import torch

from algorithms.dfot.backbones.u_vit.u_vit3d_pose import PhysicalTransitionConditioner
from upcast.factorized_transitions import SpatialResidualVQ, WorldTokenFusion


def test_residual_quantization_has_finite_gradients():
    torch.manual_seed(7)
    module = SpatialResidualVQ(
        input_dim=32, code_dim=8, num_quantizers=2, codebook_size=16,
        dead_code_threshold=0,
    )
    tokens = torch.randn(2, 4, 32, requires_grad=True)
    quantized, commitment, metrics = module(tokens)
    assert metrics["indices"].shape == (2, 4, 2)
    (quantized.square().mean() + commitment).backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_additive_coupling_is_invertible():
    torch.manual_seed(11)
    module = WorldTokenFusion(16)
    physical = torch.randn(2, 4, 16, requires_grad=True)
    appearance = torch.randn_like(physical, requires_grad=True)
    world = module(physical, appearance)
    recovered_physical, recovered_appearance = module.inverse(world)
    assert world.shape == (2, 4, 32)
    assert torch.allclose(recovered_physical, physical, atol=1e-6)
    assert torch.allclose(recovered_appearance, appearance, atol=1e-6)
    world.square().mean().backward()
    assert torch.isfinite(physical.grad).all()
    assert torch.isfinite(appearance.grad).all()


def test_conditioner_starts_as_a_zero_residual():
    torch.manual_seed(17)
    module = PhysicalTransitionConditioner(
        hidden_dim=16, physical_dim=8, num_heads=4, spatial_grid=2,
        max_read_scale=0.05,
    )
    latent = torch.randn(2, 3, 16, 4, 4)
    physical = torch.randn(2, 4, 8)
    conditioned, velocity, gate = module(
        latent, physical, torch.tensor([1, 2]), torch.tensor([-1.0, 1.0]),
        video_physical=torch.randn_like(physical),
        fused_physical=torch.randn_like(physical),
    )
    assert torch.equal(conditioned, latent)
    assert velocity.shape == physical.shape
    assert gate.shape == (2, 4, 1)
    (velocity.square().mean() + conditioned.square().mean()).backward()
    assert torch.isfinite(module.read_gate.grad)
