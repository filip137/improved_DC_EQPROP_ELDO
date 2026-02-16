#!/usr/bin/env bash
set -euo pipefail

run_moons() {
  local run_dir="${1:-}"
  if [[ -z "$run_dir" ]]; then
    echo "usage: run_moons /path/to/run_dir" >&2
    return 2
  fi

  run_dir="${run_dir%/}"
  local inputs="${run_dir}/linspace_states.npz"
  local weights="${run_dir}/model.npz"
  local config="${run_dir}/run_metadata.json"

  python3 validate_cd_results.py --moons \
    --inputs "$inputs" \
    --weights "$weights" \
    --config "$config"
}
