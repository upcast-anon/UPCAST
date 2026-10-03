from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("recipe,steps", [
    ("acquire", 1000), ("rollout_adapter", 200), ("selective_joint", 250),
])
def test_training_recipes_compose_with_public_schema(recipe, steps):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configurations")):
        cfg = compose(config_name="config", overrides=[
            "dataset=realestate10k", "algorithm=dfot_upcast", f"recipe={recipe}",
        ])
    assert cfg.experiment.training.max_steps == steps
    assert cfg.algorithm.alignment.unit_arch_version == 7
    assert cfg.algorithm.alignment.unit_migrate_from_version is None
    assert cfg.wandb.entity is None
    assert cfg.wandb.mode == "offline"
    assert cfg.load.startswith("checkpoints/")
    assert cfg.algorithm.backbone.world_state.enable
    assert cfg.algorithm.alignment.unit_inference_enable == (recipe != "acquire")
    assert OmegaConf.to_container(cfg, resolve=True)


def test_public_imports_resolve_inside_the_release():
    from algorithms.dfot import DFoTUPCAST, DFoTActionUPCAST
    from datasets.video.arkitscenes import ARKitScenesAdvancedVideoDataset
    from datasets.video.rgbd import RGBDSensorVideoDataset
    import upcast.factorized_transitions as transitions
    import upcast.transition_learning as learning

    assert DFoTUPCAST.__name__ == "DFoTUPCAST"
    assert DFoTActionUPCAST.__name__ == "DFoTActionUPCAST"
    assert issubclass(ARKitScenesAdvancedVideoDataset, RGBDSensorVideoDataset)
    assert Path(transitions.__file__).resolve().is_relative_to(ROOT)
    assert Path(learning.__file__).resolve().is_relative_to(ROOT)


@pytest.mark.parametrize("frames", [64, 256])
def test_evaluation_entry_disables_auxiliary_inference(tmp_path, frames):
    interpreter = tmp_path / "capture_args"
    interpreter.write_text(
        "#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    interpreter.chmod(0o755)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test fixture")
    env = os.environ.copy()
    env.update({
        "PYTHON": str(interpreter), "EVAL_CKPT": str(checkpoint),
        "RESULT_DIR": str(tmp_path / "result"), "SAVE_RAW": "false",
        "N_FRAMES": str(frames), "CONTEXT_LENGTH": "1", "FRAME_SKIP": "1",
        "EVAL_GENERATIVE_UNIT_ENABLE": "false", "WANDB_MODE": "offline",
    })
    result = subprocess.run(
        ["bash", "scripts/eval_upcast.sh"], cwd=ROOT, env=env,
        capture_output=True, text=True, check=True,
    )
    args = json.loads(result.stdout.splitlines()[-1])
    for setting in (
        "algorithm=dfot_upcast", "dataset.context_length=1", "dataset.frame_skip=1",
        "algorithm.backbone.generative_unit.enable=false",
        "algorithm.backbone.world_state.enable=false",
        "algorithm.alignment.unit_inference_enable=false",
        "algorithm.alignment.generative_unit_mode=off",
        "wandb.mode=offline",
    ):
        assert setting in args
    assert f"dataset.n_frames={frames}" in args


def test_offline_run_does_not_require_a_wandb_account(monkeypatch):
    import main

    cfg = OmegaConf.create({
        "name": "UPCAST_test", "wandb": {"entity": None, "project": "upcast", "mode": "offline"},
        "resume": None, "load": None, "skip_download": True,
    })
    calls = []
    monkeypatch.setattr(main, "run_local", lambda cfg: calls.append(cfg))
    main.run.__wrapped__(cfg)
    assert calls == [cfg]


def test_cloud_checkpoint_requires_the_users_wandb_account(monkeypatch):
    import main

    cfg = OmegaConf.create({
        "name": "UPCAST_test", "wandb": {"entity": None, "project": "upcast", "mode": "offline"},
        "resume": "test_run", "load": None, "skip_download": True,
    })
    with pytest.raises(ValueError, match="wandb.entity"):
        main.run.__wrapped__(cfg)
