#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

clone_once() {
  local url="$1" destination="$2"
  if [[ -d "${destination}" ]]; then
    printf 'Keeping existing dependency: %s\n' "${destination}"
  else
    git clone --depth 1 "${url}" "${destination}"
  fi
}

clone_once https://github.com/kaichen-z/PAGE4D.git external/page4d
clone_once https://github.com/facebookresearch/jepa.git external/jepa
clone_once https://github.com/oooolga/JEDi.git external/JEDi

printf 'Source dependencies are ready. Download their weights separately.\n'
