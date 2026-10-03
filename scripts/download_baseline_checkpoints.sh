#!/usr/bin/env bash
set -euo pipefail

# Public baseline weights; UPCAST weights are supplied separately.
mkdir -p checkpoints
hf download Haoyuwu/GeometryForcing --include "DFoT_16f_state_dict.ckpt" --local-dir ./checkpoints
hf download Haoyuwu/GeometryForcing --include "geometry_forcing_state_dict.ckpt" --local-dir ./checkpoints
