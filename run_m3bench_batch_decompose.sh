#!/bin/bash
# ============================================================
# M3-Bench batch-level self-evolution evaluation script (decompose_only control experiment)
#
# Key differences from run_m3bench_batch.sh:
#   1. AGENT defaults to decompose_only (keeps task decomposition + focus scheduling,
#      but injects no ledger content into the context; used as the control experiment
#      for ledger_multi_round)
#   2. AGENT_TAG = agdecomp; the output directory is fully isolated from run_m3bench_batch.sh / *_agledger
#   3. Other parameters (dataset, model, batch_size, self-evolution, visual layer, etc.) default
#      to exactly the same values as run_m3bench_batch.sh, and can be overridden via environment variables
#
# Usage examples:
#   bash run_m3bench_batch_decompose.sh
#   NUM_QUESTIONS=128 BATCH_SIZE=64 bash run_m3bench_batch_decompose.sh
#   AGENT=ledger_multi_round bash run_m3bench_batch_decompose.sh   # switching back to ledger is also supported
# ============================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

LV_HARNESS_PATH="/path/to/lv_harness"
export PYTHONPATH="${LV_HARNESS_PATH}"

# ---- Run mode ----
MODE="${MODE:-batch}"

# ---- Dataset ----
DATASET="${DATASET:-web}"
M3_ANNOTATIONS_ROOT="${M3_ANNOTATIONS_ROOT:-/path/to/m3-agent/data/annotations}"
M3_EVENTGRAPH_ROOT="${M3_EVENTGRAPH_ROOT:-/path/to/data/M3-Bench/event_graph_clip30}"
M3_VIDEOS_ROOT="${M3_VIDEOS_ROOT:-/path/to/data/M3-Bench/videos}"
M3_CLIPS_ROOT="${M3_CLIPS_ROOT:-/path/to/data/M3-Bench/clips}"

ANNOTATION_FILE="${ANNOTATION_FILE:-${M3_ANNOTATIONS_ROOT}/${DATASET}.json}"
EVENTGRAPH_DIR="${EVENTGRAPH_DIR:-${M3_EVENTGRAPH_ROOT}/${DATASET}}"

# ---- Tunable parameters ----
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
MODEL="${MODEL:-gemini-2.5-flash}"
BACKEND="${BACKEND:-openai}"
WORKERS="${WORKERS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STRATEGY="${STRATEGY:-hierarchical}"
SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-true}"
SEED="${SEED:-5}"

# ---- Inference Agent (defaults to decompose_only) ----
# Available values: multi_round_search | ledger_multi_round | decompose_only
AGENT="${AGENT:-decompose_only}"
LEDGER_MAX_SUBTASKS="${LEDGER_MAX_SUBTASKS:-5}"
LEDGER_MAX_ATTEMPTS="${LEDGER_MAX_ATTEMPTS:-4}"
LEDGER_SYNTHESIS_RAW="${LEDGER_SYNTHESIS_RAW:-false}"

# ---- Evaluation ----
EVAL_MODEL="${EVAL_MODEL:-gemini-2.5-flash}"
USE_STRING_MATCH="${USE_STRING_MATCH:-false}"

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
SUBSET_OUTPUT="${SUBSET_OUTPUT:-${LV_HARNESS_PATH}/data/annotations/m3bench_${DATASET}_${NUM_SUFFIX}.json}"

# ---- Output directory ----
if [ "$NUM_QUESTIONS" = "-1" ]; then
    N_SUFFIX="all"
else
    N_SUFFIX="n${NUM_QUESTIONS}"
fi

if [ "${ENABLE_VISUAL_LAYER:-true}" = "true" ]; then
    KF_TAG="kfon"
else
    KF_TAG="kfoff"
fi
if [ "${ENABLE_EVOLUTION:-true}" = "true" ]; then
    EVO_TAG="evoon"
else
    EVO_TAG="evooff"
fi

# Agent tag: agbase / agledger / agdecomp
case "${AGENT}" in
    "ledger_multi_round")
        AGENT_TAG="agledger"
        ;;
    "decompose_only")
        AGENT_TAG="agdecomp"
        ;;
    *)
        AGENT_TAG="agbase"
        ;;
esac

EVO_MODE_TAG="${EVO_MODE_TAG:-batchevo}"

RESULTS_ROOT="${LV_HARNESS_PATH}/data/results"
DIR_PREFIX="m3bench_${DATASET}_${N_SUFFIX}_${KF_TAG}_${EVO_TAG}_${EVO_MODE_TAG}_${AGENT_TAG}_${MODEL}_"
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

# ---- Visual layer (KEYFRAME) ----
ENABLE_VISUAL_LAYER="${ENABLE_VISUAL_LAYER:-true}"
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
SKILL_INSTRUCTIONS_MODEL="${SKILL_INSTRUCTIONS_MODEL:-${MODEL}}"
# Note: EVOLUTION_DIR is isolated by AGENT_TAG by default to avoid mixing data with the agledger experiment
EVOLUTION_DIR="${EVOLUTION_DIR:-${LV_HARNESS_PATH}/data/evolution/m3bench_${DATASET}_${EVOLUTION_SUFFIX}_${KF_TAG}_${EVO_MODE_TAG}_${AGENT_TAG}_${SKILL_INSTRUCTIONS_MODEL}}"
LOAD_PRIOR_SKILLS="${LOAD_PRIOR_SKILLS:-false}"
LOAD_PRIOR_LEARNINGS="${LOAD_PRIOR_LEARNINGS:-false}"
PROMOTE_THRESHOLD="${PROMOTE_THRESHOLD:-8}"
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
    --seed ${SEED}"

# Decompose / Ledger shared Planner+Scheduler parameters
if [ "${AGENT}" = "ledger_multi_round" ] || [ "${AGENT}" = "decompose_only" ]; then
    CMD="${CMD} --ledger_max_subtasks ${LEDGER_MAX_SUBTASKS}"
    CMD="${CMD} --ledger_max_attempts ${LEDGER_MAX_ATTEMPTS}"
fi

# Ledger-only: synthesis raw debug switch
if [ "${AGENT}" = "ledger_multi_round" ] && [ "${LEDGER_SYNTHESIS_RAW}" = "true" ]; then
    CMD="${CMD} --ledger_synthesis_raw"
fi

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

if [ "${ENABLE_EVOLUTION}" = "true" ]; then
    CMD="${CMD} --evolution --evolution_dir ${EVOLUTION_DIR}"
    CMD="${CMD} --promote_threshold ${PROMOTE_THRESHOLD}"
    CMD="${CMD} --route_threshold ${ROUTE_THRESHOLD}"
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
echo "M3-Bench batch-level self-evolution evaluation [decompose-only control] (dataset=${DATASET}, mode=${MODE})"
echo "  ENTRY           : run_homer_batch_evo.py"
echo "  ANNOTATION_FILE : ${ANNOTATION_FILE}"
echo "  EVENTGRAPH_DIR  : ${EVENTGRAPH_DIR}"
echo "  CLIP_DIR_TEMPLATE: ${CLIP_DIR_TEMPLATE}"
echo "  NUM_QUESTIONS   : ${NUM_QUESTIONS}"
echo "  MODEL           : ${MODEL}"
echo "  AGENT           : ${AGENT}    (TAG=${AGENT_TAG})"
echo "  TAGS            : ${KF_TAG} / ${EVO_TAG} / ${EVO_MODE_TAG} / ${AGENT_TAG}"
echo "  SEED            : ${SEED}"
echo "  BATCH_SIZE      : ${BATCH_SIZE}"
echo "  EVAL_MODEL      : ${EVAL_MODEL}"
echo "  USE_STRING_MATCH: ${USE_STRING_MATCH}"
echo "  OUTPUT_DIR      : ${OUTPUT_DIR}"
if [ "${ENABLE_VISUAL_LAYER}" = "true" ]; then
    echo "  VISUAL_LAYER    : enabled"
    echo "    CLIPS_ROOT    : ${VISUAL_CLIPS_ROOT}"
    echo "    KEYFRAME_DIR  : ${KEYFRAME_DIR_NAME}"
    echo "    MAX_IMAGES    : ${MAX_IMAGES_PER_CALL}"
else
    echo "  VISUAL_LAYER    : disabled"
fi
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
