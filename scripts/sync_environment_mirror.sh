#!/usr/bin/env bash
# Install the exact uv.lock versions/hashes through a chosen package mirror.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
posttrain_uv="${UV_BIN:-uv}"
posttrain_python="${POSTTRAIN_BOOTSTRAP_PYTHON:-python3.12}"
posttrain_environment="${UV_PROJECT_ENVIRONMENT:-.venv}"
posttrain_index="${POSTTRAIN_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
posttrain_requirements="$(mktemp)"
trap 'rm -f -- "$posttrain_requirements"' EXIT
"$posttrain_uv" venv --allow-existing --python "$posttrain_python" "$posttrain_environment"
"$posttrain_uv" export --frozen --all-groups --no-emit-project --format requirements-txt \
  --output-file "$posttrain_requirements" --quiet
"$posttrain_uv" pip sync --python "$posttrain_environment/bin/python" \
  --index-url "$posttrain_index" --require-hashes "$posttrain_requirements"
"$posttrain_uv" pip install --python "$posttrain_environment/bin/python" \
  --index-url "$posttrain_index" --no-deps --editable .
