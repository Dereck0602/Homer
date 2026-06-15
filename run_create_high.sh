#!/bin/bash

# ============================================================
# Usage examples:
#   bash run_create_high.sh                          # use default parameters
#   bash run_create_high.sh --model gemini-3-flash-preview
#   bash run_create_high.sh --folder_path /path/to/pkl --out_dir /path/to/output --model DeepSeek-V3.1T --max_workers 4 --batch_size 30
# ============================================================

# ---------- Default parameters (modify as needed) ----------
FOLDER_PATH="/path/to/data/M3-Bench/memory_graphs/web"
OUT_DIR="/path/to/data/M3-Bench/event_graph_clip10/web"
#FOLDER_PATH="/path/to/data/LVOmniBench/videos/memory_graphs"
#OUT_DIR="/path/to/data/LVOmniBench/videos/event_graphs"
#MODEL="m3-memory"
MODEL="gemini-3-flash-preview"
MAX_WORKERS=4
BATCH_SIZE=10
API_CONFIG="configs/api_config.json"  # leave empty to use the Python-side default (configs/api_config.json)

# ---------- Parse command-line arguments (override defaults) ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --folder_path) FOLDER_PATH="$2"; shift 2 ;;
        --out_dir)     OUT_DIR="$2";     shift 2 ;;
        --model)       MODEL="$2";       shift 2 ;;
        --max_workers) MAX_WORKERS="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --api_config)  API_CONFIG="$2";  shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "========== Parameter configuration =========="
echo "  folder_path : ${FOLDER_PATH}"
echo "  out_dir     : ${OUT_DIR}"
echo "  model       : ${MODEL}"
echo "  max_workers : ${MAX_WORKERS}"
echo "  batch_size  : ${BATCH_SIZE}"
echo "  api_config  : ${API_CONFIG:-<default>}"
echo "=============================="

# Assemble optional arguments
OPTIONAL_ARGS=""
if [[ -n "${API_CONFIG}" ]]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --api_config ${API_CONFIG}"
fi

python create_high.py \
    --folder_path "${FOLDER_PATH}" \
    --out_dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --max_workers "${MAX_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    ${OPTIONAL_ARGS}
