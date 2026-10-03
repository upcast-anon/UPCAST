from pathlib import Path

import torch

from evaluation.export_inference_checkpoint import export_checkpoint


def test_export_keeps_only_diffusion_values_and_drops_optimizer(tmp_path: Path):
    source = tmp_path / "training.ckpt"
    output = tmp_path / "inference.ckpt"
    diffusion = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(
        {
            "global_step": 17,
            "state_dict": {
                "diffusion_model.block.weight": diffusion,
                "representation_alignment.projector.weight": torch.ones(2, 2),
            },
            "optimizer_states": [{"state": {0: {"exp_avg": torch.ones(3, 4)}}}],
            "lr_schedulers": [{"last_epoch": 17}],
            "spatial_unit": {"latent_return_blocks": [-2]},
        },
        source,
    )

    report = export_checkpoint(source, output, "diffusion_model.")
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert report["global_step"] == 17
    assert report["tensor_count"] == 1
    assert report["discarded_model_tensor_count"] == 1
    assert report["discarded_metadata_keys"] == ["spatial_unit"]
    assert payload["optimizer_states"] == []
    assert "spatial_unit" not in payload
    assert list(payload["state_dict"]) == ["diffusion_model.block.weight"]
    assert torch.equal(payload["state_dict"]["diffusion_model.block.weight"], diffusion)
