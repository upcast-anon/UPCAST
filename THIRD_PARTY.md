# Third-Party Components

UPCAST adds factorized transition learning, selective consolidation, rollout
refinement, and their evaluation tools. Upstream copyright notices are
retained; they identify dependency authors, not the anonymous UPCAST authors.

| Component | Location | Source and Terms |
| --- | --- | --- |
| Research template / DFoT foundation | `algorithms/`, `datasets/`, `experiments/`, `utils/` | [Research template](https://github.com/buoyancy99/research-template), [DFoT](https://github.com/kwsong0113/diffusion-forcing-transformer); original MIT notices in `LICENSE` |
| GeometryForcing foundation | Backbone, camera conditioning, and upstream evaluation | [GeometryForcing](https://github.com/CIntellifusion/GeometryForcing); Apache 2.0 notices in `LICENSE` and `NOTICE` |
| VGGT source snapshot | `external/vggt/` | [Original VGGT release](https://github.com/facebookresearch/vggt/tree/40eb1c5cf31ad0e6f63c9a9ba97aa5e7a872c86b); CC BY-NC 4.0 in `licenses/VGGT.txt`. Local changes include DFoT feature capture and import paths. |
| DROID-SLAM evaluation source | `evaluation/reprojection_error/droid_slam/` | [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM); BSD 3-Clause in `licenses/DROID-SLAM.txt`. Local integration provides generated-video evaluation. |

The full Apache 2.0 text is provided in `licenses/Apache-2.0.txt`. UPCAST
additions are distributed under Apache 2.0; third-party components retain their
own terms. This is not a blanket license for pretrained weights or datasets.

The setup scripts download PAGE-4D, DINOv2, V-JEPA, VideoREPA, JEDi, and
DROID-SLAM assets from their upstream repositories. Those assets are not
bundled, and their respective source, weight, and dataset terms apply.
VGGT's current weight license differs from the license of the historical code
snapshot; consult the upstream model card before downloading or using weights.
