#!/bin/bash
# ============================================================
# M3-Bench batch-level evaluation script (control_api_harness mode)
#
# Wraps the inference logic of control_api.py into an lv_harness Agent,
# keeping the original prompt and retrieval method unchanged, while adding
# harness mechanisms:
#   - Guardrails: format validation + self-repair + duplicate query detection
#   - Sufficiency: information sufficiency assessment + early-stop signal
#   - Budget: consecutive empty retrievals / token budget exceeded -> force final round
#   - ConversationManager: context window management
#   - Evolution: self-evolution (learning capture + skill promote)
#
# Usage examples:
#   bash run_m3bench_batch_control_harness.sh
#   DATASET=robot NUM_QUESTIONS=128 bash run_m3bench_batch_control_harness.sh
# ============================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1

LV_HARNESS_PATH="/path/to/lv_harness"
export PYTHONPATH="${LV_HARNESS_PATH}"

# ---- Run mode ----
MODE="${MODE:-batch}"

# ---- Dataset ----
DATASET="${DATASET:-robot}"

# Route to different data paths and evaluation methods based on DATASET
if [ "${DATASET}" = "videomme" ]; then
    # Video-MME dataset (multiple choice)
    VIDEOMME_DATA_ROOT="${VIDEOMME_DATA_ROOT:-/path/to/data/Video-MME/data}"
    ANNOTATION_FILE="${ANNOTATION_FILE:-/path/to/m3-agent/data/annotations/videomme.json}"
    EVENTGRAPH_DIR="${EVENTGRAPH_DIR:-${VIDEOMME_DATA_ROOT}/event_graphs}"
    USE_STRING_MATCH="${USE_STRING_MATCH:-true}"
else
    # M3-Bench dataset (open-ended QA: robot / web etc.)
    M3_ANNOTATIONS_ROOT="${M3_ANNOTATIONS_ROOT:-/path/to/m3-agent/data/annotations}"
    M3_EVENTGRAPH_ROOT="${M3_EVENTGRAPH_ROOT:-/path/to/data/M3-Bench/event_graph_clip30}"
    M3_VIDEOS_ROOT="${M3_VIDEOS_ROOT:-/path/to/data/M3-Bench/videos}"
    M3_CLIPS_ROOT="${M3_CLIPS_ROOT:-/path/to/data/M3-Bench/clips}"
    ANNOTATION_FILE="${ANNOTATION_FILE:-${M3_ANNOTATIONS_ROOT}/${DATASET}.json}"
    EVENTGRAPH_DIR="${EVENTGRAPH_DIR:-${M3_EVENTGRAPH_ROOT}/${DATASET}/refined}"
    USE_STRING_MATCH="${USE_STRING_MATCH:-false}"
fi

# ---- Tunable parameters ----
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
MODEL="${MODEL:-m3-control}"
BACKEND="${BACKEND:-openai}"
WORKERS="${WORKERS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STRATEGY="${STRATEGY:-hierarchical}"
SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-true}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.6}"

# ---- Inference Agent (fixed to control_api_harness) ----
AGENT="${AGENT:-control_api_harness}"

# ---- Evaluation ----
EVAL_MODEL="${EVAL_MODEL:-gemini-2.5-flash}"
# USE_STRING_MATCH is automatically set based on DATASET in the dataset routing (videomme=true, m3bench=false)

# ---- DeepSeek-V4 Thinking ----
ENABLE_THINKING="${ENABLE_THINKING:-false}"

# ---- Paths ----
API_CONFIG="${API_CONFIG:-${LV_HARNESS_PATH}/configs/api_config.json}"
MEMORY_CONFIG="${MEMORY_CONFIG:-${LV_HARNESS_PATH}/configs/memory_config.json}"

# ---- Subset JSON output ----
if [ "$NUM_QUESTIONS" = "-1" ]; then
    NUM_SUFFIX="all"
else
    NUM_SUFFIX="subset_${NUM_QUESTIONS}"
fi
if [ "${DATASET}" = "videomme" ]; then
    SUBSET_OUTPUT="${SUBSET_OUTPUT:-${LV_HARNESS_PATH}/data/annotations/videomme_${NUM_SUFFIX}.json}"
else
    SUBSET_OUTPUT="${SUBSET_OUTPUT:-${LV_HARNESS_PATH}/data/annotations/m3bench_${DATASET}_${NUM_SUFFIX}.json}"
fi

# ---- Output directory ----
if [ "$NUM_QUESTIONS" = "-1" ]; then
    N_SUFFIX="all"
else
    N_SUFFIX="n${NUM_QUESTIONS}"
fi

# ---- Switch tags ----
# control_api_harness does not use the visual layer (kept consistent with the original control_api.py)
ENABLE_VISUAL_LAYER="${ENABLE_VISUAL_LAYER:-false}"
if [ "${ENABLE_VISUAL_LAYER}" = "true" ]; then
    KF_TAG="kfon"
else
    KF_TAG="kfoff"
fi
if [ "${ENABLE_EVOLUTION:-true}" = "true" ]; then
    EVO_TAG="evoon"
else
    EVO_TAG="evooff"
fi

AGENT_TAG="agctrl"
EVO_MODE_TAG="${EVO_MODE_TAG:-batchevo}"

RESULTS_ROOT="${LV_HARNESS_PATH}/data/results"
if [ "${DATASET}" = "videomme" ]; then
    DIR_PREFIX="videomme_${N_SUFFIX}_${KF_TAG}_${EVO_TAG}_${EVO_MODE_TAG}_${AGENT_TAG}_${MODEL}_"
else
    DIR_PREFIX="m3bench_${DATASET}_${N_SUFFIX}_${KF_TAG}_${EVO_TAG}_${EVO_MODE_TAG}_${AGENT_TAG}_${MODEL}_"
fi
AUTO_RESUME="${AUTO_RESUME:-false}"

if [ -n "${RESUME_DIR:-}" ]; then
    OUTPUT_DIR="${RESUME_DIR}"
    echo "[RESUME] Resuming with the specified directory: ${OUTPUT_DIR}"
elif [ "${AUTO_RESUME}" = "true" ]; then
    LATEST_DIR=$(ls -1dt "${RESULTS_ROOT}/${DIR_PREFIX}"* 2>/dev/null | head -n 1)
    if [ -n "${LATEST_DIR}" ] && [ -d "${LATEST_DIR}" ]; then
        OUTPUT_DIR="${LATEST_DIR}"
        echo "[AUTO_RESUME] Found the most recent directory to resume: ${OUTPUT_DIR}"
    else
        OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${DIR_PREFIX}$(date +%Y%m%d_%H%M%S)}"
        echo "[AUTO_RESUME] No resumable directory found, creating new one: ${OUTPUT_DIR}"
    fi
else
    OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${DIR_PREFIX}$(date +%Y%m%d_%H%M%S)}"
fi

# ---- Streaming inference parameters ----
CLIP_DIR_TEMPLATE="${CLIP_DIR_TEMPLATE:-${M3_CLIPS_ROOT}/${DATASET}/{video_name}}"
EG_INCREMENTAL="${EG_INCREMENTAL:-true}"
EG_UPDATE_INTERVAL="${EG_UPDATE_INTERVAL:-10}"
EG_MODEL="${EG_MODEL:-gemini-3-flash-preview}"
MEMORY_CACHE_DIR="${MEMORY_CACHE_DIR:-${LV_HARNESS_PATH}/data/m3bench_${DATASET}_memory_cache}"

# ---- Visual layer (disabled by default, kept consistent with control_api.py) ----
VISUAL_CLIPS_ROOT="${VISUAL_CLIPS_ROOT:-${M3_VIDEOS_ROOT}/${DATASET}/clips}"
KEYFRAME_DIR_NAME="${KEYFRAME_DIR_NAME:-keyframe_hybridv2}"
MAX_IMAGES_PER_CALL="${MAX_IMAGES_PER_CALL:-5}"

# ---- Self-evolution ----
ENABLE_EVOLUTION="${ENABLE_EVOLUTION:-true}"
if [ "$NUM_QUESTIONS" = "-1" ]; then
    EVOLUTION_SUFFIX="all"
else
    EVOLUTION_SUFFIX="n${NUM_QUESTIONS}"
fi
WISDOM_USE_LLM="${WISDOM_USE_LLM:-true}"
WISDOM_MODEL="${WISDOM_MODEL:-gemini-2.5-flash}"
REFLECTION_MAX_TOKENS="${REFLECTION_MAX_TOKENS:-8192}"
SKILL_USE_LLM_INSTRUCTIONS="${SKILL_USE_LLM_INSTRUCTIONS:-true}"
SKILL_INSTRUCTIONS_MODEL="${SKILL_INSTRUCTIONS_MODEL:-gemini-3-flash-preview}"
if [ "${DATASET}" = "videomme" ]; then
    EVOLUTION_DIR="${EVOLUTION_DIR:-${LV_HARNESS_PATH}/data/evolution/videomme_all___gemini-3-flash-preview}"
else
    EVOLUTION_DIR="${EVOLUTION_DIR:-${LV_HARNESS_PATH}/data/evolution/m3bench_${DATASET}_${EVOLUTION_SUFFIX}_${KF_TAG}_${EVO_MODE_TAG}_${AGENT_TAG}_${SKILL_INSTRUCTIONS_MODEL}}"
fi
LOAD_PRIOR_SKILLS="${LOAD_PRIOR_SKILLS:-false}"
LOAD_PRIOR_LEARNINGS="${LOAD_PRIOR_LEARNINGS:-false}"
PROMOTE_THRESHOLD="${PROMOTE_THRESHOLD:-9999999}"
ROUTE_THRESHOLD="${ROUTE_THRESHOLD:-0.45}"

# ---- Build command ----
CMD="python ${LV_HARNESS_PATH}/run_homer_batch_evo.py \
    --annotation_file ${ANNOTATION_FILE} \
    --subset_output ${SUBSET_OUTPUT} \
    --num_questions ${NUM_QUESTIONS} \
    --eventgraph_dir ${EVENTGRAPH_DIR} \
    --memory_config ${MEMORY_CONFIG} \
    --strategy ${STRATEGY} \
    --model ${MODEL} \
    --backend ${BACKEND} \
    --max_rounds ${MAX_ROUNDS} \
    --workers ${WORKERS} \
    --batch_size ${BATCH_SIZE} \
    --api_config ${API_CONFIG} \
    --eval_model ${EVAL_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --agent ${AGENT} \
    --temperature ${TEMPERATURE} \
    --seed ${SEED}"

# Force scoring through eval_model
if [ "${USE_STRING_MATCH}" = "true" ]; then
    CMD="${CMD} --string_match_first"
else
    CMD="${CMD} --no_string_match"
fi

if [ "${SAVE_TRAJECTORY}" = "true" ]; then
    CMD="${CMD} --save_trajectory"
fi

if [ "${ENABLE_THINKING}" = "true" ]; then
    CMD="${CMD} --enable_thinking"
fi

# Visual layer (disabled by default)
if [ "${ENABLE_VISUAL_LAYER}" = "true" ]; then
    CMD="${CMD} --visual_layer_enabled"
    CMD="${CMD} --visual_clips_root ${VISUAL_CLIPS_ROOT}"
    CMD="${CMD} --visual_keyframe_dir_name ${KEYFRAME_DIR_NAME}"
    CMD="${CMD} --visual_max_images_per_call ${MAX_IMAGES_PER_CALL}"
fi

if [ "${MODE}" = "streaming" ]; then
    CMD="${CMD} --streaming"
    CMD="${CMD} --clip_dir_template ${CLIP_DIR_TEMPLATE}"
    CMD="${CMD} --eventgraph_update_interval ${EG_UPDATE_INTERVAL}"
    CMD="${CMD} --eventgraph_model ${EG_MODEL}"
    if [ "${EG_INCREMENTAL}" = "true" ]; then
        CMD="${CMD} --eventgraph_incremental"
    fi
    if [ -n "${MEMORY_CACHE_DIR}" ]; then
        CMD="${CMD} --memory_cache_dir ${MEMORY_CACHE_DIR}"
    fi
fi

# ---- Self-evolution ----
EVOLUTION_READONLY="${EVOLUTION_READONLY:-true}"
if [ "${ENABLE_EVOLUTION}" = "true" ]; then
    CMD="${CMD} --evolution --evolution_dir ${EVOLUTION_DIR}"
    CMD="${CMD} --promote_threshold ${PROMOTE_THRESHOLD}"
    CMD="${CMD} --route_threshold ${ROUTE_THRESHOLD}"
    if [ "${EVOLUTION_READONLY}" = "true" ]; then
        CMD="${CMD} --evolution_readonly"
    fi
    if [ "${LOAD_PRIOR_SKILLS}" = "true" ]; then
        CMD="${CMD} --load_prior_skills"
    fi
    if [ "${LOAD_PRIOR_LEARNINGS}" = "true" ]; then
        CMD="${CMD} --load_prior_learnings"
    fi
    if [ "${WISDOM_USE_LLM}" = "true" ]; then
        CMD="${CMD} --wisdom_use_llm --wisdom_model ${WISDOM_MODEL}"
        CMD="${CMD} --reflection_llm_max_tokens ${REFLECTION_MAX_TOKENS}"
    fi
    if [ "${SKILL_USE_LLM_INSTRUCTIONS}" = "true" ]; then
        CMD="${CMD} --skill_use_llm_instructions --skill_instructions_llm_model ${SKILL_INSTRUCTIONS_MODEL}"
    fi
fi

# ---- Run ----
echo "============================================================"
echo "Batch-level evaluation [control_api_harness] (dataset=${DATASET}, mode=${MODE})"
echo "  ENTRY           : run_homer_batch_evo.py"
echo "  ANNOTATION_FILE : ${ANNOTATION_FILE}"
echo "  EVENTGRAPH_DIR  : ${EVENTGRAPH_DIR}"
echo "  NUM_QUESTIONS   : ${NUM_QUESTIONS}"
echo "  MODEL           : ${MODEL}"
echo "  AGENT           : ${AGENT}    (TAG=${AGENT_TAG})"
echo "  MAX_ROUNDS      : ${MAX_ROUNDS}"
echo "  TEMPERATURE     : ${TEMPERATURE}"
echo "  SEED            : ${SEED}"
echo "  BATCH_SIZE      : ${BATCH_SIZE}"
echo "  EVAL_MODEL      : ${EVAL_MODEL}"
echo "  USE_STRING_MATCH: ${USE_STRING_MATCH}"
echo "  OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "  VISUAL_LAYER    : ${ENABLE_VISUAL_LAYER}"
if [ -d "${OUTPUT_DIR}" ]; then
    DONE_COUNT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "*-lv_harness.jsonl" -exec wc -l {} + 2>/dev/null | tail -n 1 | awk '{print $1}')
    echo "  [RESUME INFO]   : Directory already exists, ${DONE_COUNT:-0} records completed, will be skipped automatically"
fi
if [ "${ENABLE_EVOLUTION}" = "true" ]; then
    echo "  EVOLUTION       : enabled"
    echo "    MODE          : batch-level skill promote"
    echo "    EVOLUTION_DIR : ${EVOLUTION_DIR}"
    echo "    LOAD_SKILLS   : ${LOAD_PRIOR_SKILLS}"
    echo "    LOAD_LEARNINGS: ${LOAD_PRIOR_LEARNINGS}"
    echo "    PROMOTE_THRES : ${PROMOTE_THRESHOLD}"
    echo "    ROUTE_THRESH  : ${ROUTE_THRESHOLD}"
    echo "    REFL_MAX_TOK  : ${REFLECTION_MAX_TOKENS}"
else
    echo "  EVOLUTION       : disabled"
fi
echo "============================================================"
eval ${CMD}
