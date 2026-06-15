#!/bin/bash
# ============================================================
# LV-Harness evaluation script
# Mirrors m3-agent/run_eval_high.sh, run using the lv_harness framework
# ============================================================

export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=/path/to/m3-agent:$(dirname "$0")

python -m lv_harness run \
    --config configs/tasks/videomme_streaming.yaml \
    --data_file data/annotations/videomme.json \
    --eventgraph_dir /path/to/data/Video-MME/data/event_window_clip20_dpsk_new \
    --backend openai \
    --model gemini-2.5-flash \
    --workers 4 \
    --batch_mode
