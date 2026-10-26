#!/bin/bash
# Activate only the repository-local ESP-IDF installation, then run idf.py.
# Usage: bash scripts/esp32_idf.sh -C firmware/esp32_s3_smoke build
set -e
EDGEAI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export IDF_TOOLS_PATH="$EDGEAI_ROOT/artifacts/esp32-toolchain/tools"
export IDF_PYTHON_ENV_PATH="$IDF_TOOLS_PATH/python_env/idf5.5_py3.9_env"
export PATH="$IDF_PYTHON_ENV_PATH/bin:$EDGEAI_ROOT/artifacts/esp32-toolchain/bootstrap-python/bin:$PATH"
if [ ! -f "$EDGEAI_ROOT/artifacts/esp32-toolchain/esp-idf-v5.5.1/export.sh" ] || [ ! -x "$IDF_PYTHON_ENV_PATH/bin/python" ]; then
    echo "Repository-local ESP-IDF 5.5.1 is not installed; see the board bring-up documentation." >&2
    exit 1
fi
. "$EDGEAI_ROOT/artifacts/esp32-toolchain/esp-idf-v5.5.1/export.sh" >&2
exec python "$IDF_PATH/tools/idf.py" "$@"
