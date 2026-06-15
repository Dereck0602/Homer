#!/bin/bash
# ============================================================
# M3-Bench batch-level self-evolution evaluation script (reuses the lv_harness framework)
#
# Largely identical to run_m3bench_test.sh.
# Key differences:
#   1. The Python entry point is changed to run_homer_batch_evo.py
#   2. In offline batch mode, each question only captures learning; skills are promoted uniformly after each batch
#   3. The output directory and EVOLUTION_DIR add a batchevo tag by default, to avoid mixing with per-question self-evolution experiments
#
# Usage examples:
#   bash run_m3bench_batch_evo_test.sh
#   DATASET=robot NUM_QUESTIONS=128 BATCH_SIZE=64 bash run_m3bench_batch_evo_test.sh
#   MODE=streaming bash run_m3bench_batch_evo_test.sh
#
# Ablation: use VideoGraph only (without EventGraph / Keyframe / hierarchical retrieval):
#   STRATEGY=videograph_only bash run_m3bench_batch.sh
#
# Ablation: keep EventGraph node retrieval + VideoGraph, but without graph traversal and keyframes:
#   STRATEGY=no_graph_walk bash run_m3bench_batch.sh
# ============================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

LV_HARNESS_PATH="/path/to/lv_harness"
export PYTHONPATH="${LV_HARNESS_PATH}"

# ---- Run mode ----
# Allowed values: "batch" (offline batch, default) | "streaming" (streaming inference)
# Note: batch-level skill promote only applies to offline batch mode.
MODE="${MODE:-batch}"

# ---- Dataset selection (robot / web / videomme, can be overridden via environment variable) ----
DATASET="${DATASET:-robot}"
M3_ANNOTATIONS_ROOT="${M3_ANNOTATIONS_ROOT:-/path/to/m3-agent/data/annotations}"
M3_EVENTGRAPH_ROOT="${M3_EVENTGRAPH_ROOT:-/path/to/data/M3-Bench/event_graph_clip30}"
M3_VIDEOS_ROOT="${M3_VIDEOS_ROOT:-/path/to/data/M3-Bench/videos}"
M3_CLIPS_ROOT="${M3_CLIPS_ROOT:-/path/to/data/M3-Bench/clips}"

ANNOTATION_FILE="${ANNOTATION_FILE:-${M3_ANNOTATIONS_ROOT}/${DATASET}.json}"
EVENTGRAPH_DIR="${EVENTGRAPH_DIR:-${M3_EVENTGRAPH_ROOT}/${DATASET}}"

# ---- Tunable parameters ----
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
QUESTION_OFFSET="${QUESTION_OFFSET:-0}"
MODEL="${MODEL:-gemini-2.5-flash}"
BACKEND="${BACKEND:-openai}"
WORKERS="${WORKERS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STRATEGY="${STRATEGY:-hierarchical}"
# When STRATEGY=videograph_only, automatically disable keyframe and eventgraph (ablation experiment)
if [ "${STRATEGY}" = "videograph_only" ]; then
    ENABLE_VISUAL_LAYER="${ENABLE_VISUAL_LAYER:-false}"
fi
# When STRATEGY=no_graph_walk, automatically disable keyframe (do not use NEIGHBOR/KEYFRAME, keep EventGraph node retrieval + VideoGraph)
if [ "${STRATEGY}" = "no_graph_walk" ]; then
    ENABLE_VISUAL_LAYER="${ENABLE_VISUAL_LAYER:-false}"
fi
SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-true}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.8}"

# ---- Inference Agent ----
# Options: multi_round_search | ledger_multi_round
AGENT="${AGENT:-ledger_multi_round}"
LEDGER_MAX_SUBTASKS="${LEDGER_MAX_SUBTASKS:-5}"
LEDGER_MAX_ATTEMPTS="${LEDGER_MAX_ATTEMPTS:-3}"
LEDGER_SYNTHESIS_RAW="${LEDGER_SYNTHESIS_RAW:-false}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.2}"

# ---- Evaluation parameters (M3-Bench forces eval_model scoring) ----
EVAL_MODEL="${EVAL_MODEL:-gemini-2.5-flash}"
USE_STRING_MATCH="${USE_STRING_MATCH:-false}"

# ---- DeepSeek-V4 Thinking mode ----
ENABLE_THINKING="${ENABLE_THINKING:-false}"

# ---- Reason mode (whether to require the model to output a Reason line, enabled by default) ----
ENABLE_REASON="${ENABLE_REASON:-true}"

# ---- LLM decoding seed (set to -1 to not pass a seed) ----
# SEED="${SEED:-42}"  # already defined in the tunable parameters section above

# ---- Path configuration ----
API_CONFIG="${API_CONFIG:-${LV_HARNESS_PATH}/configs/api_config.json}"
MEMORY_CONFIG="${MEMORY_CONFIG:-${LV_HARNESS_PATH}/configs/memory_config.json}"

# ---- Subset JSON output ----
if [ "$NUM_QUESTIONS" = "-1" ]; then
    NUM_SUFFIX="all"
else
    NUM_SUFFIX="subset_${NUM_QUESTIONS}"
fi
# When an offset is present, add an offset marker to the file name to avoid cache conflicts
if [ "${QUESTION_OFFSET}" != "0" ]; then
    NUM_SUFFIX="${NUM_SUFFIX}_off${QUESTION_OFFSET}"
fi
SUBSET_OUTPUT="${SUBSET_OUTPUT:-${LV_HARNESS_PATH}/data/annotations/m3bench_${DATASET}_${NUM_SUFFIX}.json}"

# ---- Output directory (supports resuming from checkpoints) ----
if [ "$NUM_QUESTIONS" = "-1" ]; then
    N_SUFFIX="all"
else
    N_SUFFIX="n${NUM_QUESTIONS}"
fi

# ---- Switch tags ----
if [ "${ENABLE_VISUAL_LAYER:-true}" = "true" ]; then
    KF_TAG="kfon"
else
    KF_TAG="kfoff"
fi
# Strategy tag: used to distinguish experiments with different memory strategies
if [ "${STRATEGY}" = "videograph_only" ]; then
    STRAT_TAG="vgonly"
elif [ "${STRATEGY}" = "no_graph_walk" ]; then
    STRAT_TAG="nogwalk"
else
    STRAT_TAG="hier"
fi
if [ "${ENABLE_EVOLUTION:-true}" = "true" ]; then
    EVO_TAG="evoon"
else
    EVO_TAG="evooff"
fi

# Agent tag: agbase (multi_round_search) / agledger (ledger_multi_round)
if [ "${AGENT}" = "ledger_multi_round" ]; then
    AGENT_TAG="agledger"
else
    AGENT_TAG="agbase"
fi

# Self-evolution mode tag: batchevo means skills are promoted uniformly after each batch
EVO_MODE_TAG="${EVO_MODE_TAG:-batchevo}"

RESULTS_ROOT="${LV_HARNESS_PATH}/data/results"
DIR_PREFIX="m3bench_${DATASET}_${N_SUFFIX}_${STRAT_TAG}_${KF_TAG}_${EVO_TAG}_${EVO_MODE_TAG}_${AGENT_TAG}_${MODEL}_"
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
        echo "[AUTO_RESUME] No resumable directory found, creating a new one: ${OUTPUT_DIR}"
    fi
else
    OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${DIR_PREFIX}$(date +%Y%m%d_%H%M%S)}"
fi

# ---- Streaming inference specific parameters ----
CLIP_DIR_TEMPLATE="${CLIP_DIR_TEMPLATE:-${M3_CLIPS_ROOT}/${DATASET}/{video_name}}"
EG_INCREMENTAL="${EG_INCREMENTAL:-true}"
EG_UPDATE_INTERVAL="${EG_UPDATE_INTERVAL:-10}"
EG_MODEL="${EG_MODEL:-gemini-3-flash-preview}"
MEMORY_CACHE_DIR="${MEMORY_CACHE_DIR:-${LV_HARNESS_PATH}/data/m3bench_${DATASET}_memory_cache}"

# ---- Third layer: visual layer (KEYFRAME) ----
ENABLE_VISUAL_LAYER="${ENABLE_VISUAL_LAYER:-true}"
VISUAL_CLIPS_ROOT="${VISUAL_CLIPS_ROOT:-${M3_VIDEOS_ROOT}/${DATASET}/clips}"
KEYFRAME_DIR_NAME="${KEYFRAME_DIR_NAME:-keyframe_hybridv2}"
MAX_IMAGES_PER_CALL="${MAX_IMAGES_PER_CALL:-5}"

# ---- Self-evolution parameters ----
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
EVOLUTION_DIR="${EVOLUTION_DIR:-${LV_HARNESS_PATH}/data/evolution/m3bench_${DATASET}_${EVOLUTION_SUFFIX}_${KF_TAG}_${EVO_MODE_TAG}_${SKILL_INSTRUCTIONS_MODEL}}"
LOAD_PRIOR_SKILLS="${LOAD_PRIOR_SKILLS:-false}"
LOAD_PRIOR_LEARNINGS="${LOAD_PRIOR_LEARNINGS:-false}"
PROMOTE_THRESHOLD="${PROMOTE_THRESHOLD:-8}"
ROUTE_THRESHOLD="${ROUTE_THRESHOLD:-0.45}"

# ---- Build the command ----
CMD="python ${LV_HARNESS_PATH}/run_homer_batch_evo.py \
    --annotation_file ${ANNOTATION_FILE} \
    --subset_output ${SUBSET_OUTPUT} \
    --num_questions ${NUM_QUESTIONS} \
    --question_offset ${QUESTION_OFFSET} \
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

# TaskLedger-specific parameters (injected only for ledger-family AGENT)
if [ "${AGENT}" = "ledger_multi_round" ]; then
    CMD="${CMD} --ledger_max_subtasks ${LEDGER_MAX_SUBTASKS}"
    CMD="${CMD} --ledger_max_attempts ${LEDGER_MAX_ATTEMPTS}"
    CMD="${CMD} --planner_temperature ${PLANNER_TEMPERATURE}"
    if [ "${LEDGER_SYNTHESIS_RAW}" = "true" ]; then
        CMD="${CMD} --ledger_synthesis_raw"
    fi
fi

# Force eval_model scoring (open-ended QA)
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

if [ "${ENABLE_REASON}" = "false" ]; then
    CMD="${CMD} --no_reason"
fi

# Third layer: visual layer (KEYFRAME)
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
EVOLUTION_READONLY="${EVOLUTION_READONLY:-false}"
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
echo "M3-Bench batch-level self-evolution evaluation (dataset=${DATASET}, mode=${MODE})"
echo "  ENTRY           : run_homer_batch_evo.py"
echo "  ANNOTATION_FILE : ${ANNOTATION_FILE}"
echo "  EVENTGRAPH_DIR  : ${EVENTGRAPH_DIR}"
echo "  CLIP_DIR_TEMPLATE: ${CLIP_DIR_TEMPLATE}"
echo "  NUM_QUESTIONS   : ${NUM_QUESTIONS} (offset=${QUESTION_OFFSET})"
echo "  MODEL           : ${MODEL}"
echo "  AGENT           : ${AGENT}"
echo "  STRATEGY        : ${STRATEGY}"
echo "  TAGS            : ${STRAT_TAG} / ${KF_TAG} / ${EVO_TAG} / ${EVO_MODE_TAG} / ${AGENT_TAG}"
echo "  SEED            : ${SEED}"
echo "  TEMPERATURE     : ${TEMPERATURE}"
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
    echo "  VISUAL_LAYER    : disabled (keeps the two-layer memory behavior unchanged)"
fi
if [ -d "${OUTPUT_DIR}" ]; then
    DONE_COUNT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "*-lv_harness.jsonl" -exec wc -l {} + 2>/dev/null | tail -n 1 | awk '{print $1}')
    echo "  [RESUME INFO]   : Directory already exists, ${DONE_COUNT:-0} records completed, will skip automatically"
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
