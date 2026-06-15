#!/bin/bash
# ============================================================
# M3-Bench Skill Readonly evaluation script
#
# Function: load existing skills for inference injection, but write no learning/skill/wisdom.
# Applicable scenarios:
#   1. Pure evaluation experiment: verify the accuracy improvement from existing skills
#   2. Control experiment: compare against the baseline without skills
#   3. Avoid polluting the data in an existing evolution_dir
#
# Usage examples:
#   bash run_m3bench_batch_skill_readonly.sh
#   DATASET=web MODEL=gemini-3-flash-preview bash run_m3bench_batch_skill_readonly.sh
#   EVOLUTION_DIR=/path/to/existing/evolution_dir bash run_m3bench_batch_skill_readonly.sh
# ============================================================

# ---- Core switches: enable evolution + readonly + load existing skills ----
export ENABLE_EVOLUTION="${ENABLE_EVOLUTION:-true}"
export EVOLUTION_READONLY="${EVOLUTION_READONLY:-true}"
export LOAD_PRIOR_SKILLS="${LOAD_PRIOR_SKILLS:-true}"
export LOAD_PRIOR_LEARNINGS="${LOAD_PRIOR_LEARNINGS:-false}"

# ---- Dataset selection (can be overridden via environment variables) ----
export DATASET="${DATASET:-web}"

# ---- Model selection ----
export MODEL="${MODEL:-m3-control}"

# ---- Inference Agent ----
export AGENT="${AGENT:-ledger_multi_round}"

# ---- Other tunable parameters (inherit defaults from run_m3bench_batch.sh, can be overridden) ----
export NUM_QUESTIONS="${NUM_QUESTIONS:--1}"
export WORKERS="${WORKERS:-4}"
export MAX_ROUNDS="${MAX_ROUNDS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export TEMPERATURE="${TEMPERATURE:-0.8}"
export SEED="${SEED:-42}"

# ---- Visual layer ----
export ENABLE_VISUAL_LAYER="${ENABLE_VISUAL_LAYER:-false}"

# ---- Evolution directory (points to existing skill data) ----
# If not specified, the default path in run_m3bench_batch.sh will be used
# It is recommended to explicitly specify the directory with trained skills
export EVOLUTION_DIR="${EVOLUTION_DIR:-}"

# ---- EVO_MODE_TAG: marks the readonly mode to avoid confusion with normal self-evolution experiment directories ----
export EVO_MODE_TAG="${EVO_MODE_TAG:-skillro}"

# ---- Self-evolution write-related parameters (not actually triggered in readonly mode, but still need to be passed) ----
export WISDOM_USE_LLM="${WISDOM_USE_LLM:-false}"
export SKILL_USE_LLM_INSTRUCTIONS="${SKILL_USE_LLM_INSTRUCTIONS:-false}"
export PROMOTE_THRESHOLD="${PROMOTE_THRESHOLD:-99999}"

# ---- Resume from checkpoint ----
export AUTO_RESUME="${AUTO_RESUME:-true}"

# ---- Invoke the main script ----
LV_HARNESS_PATH="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "M3-Bench Skill Readonly evaluation"
echo "  Mode: load existing skills to inject into inference, write no updates"
echo "  DATASET         : ${DATASET}"
echo "  MODEL           : ${MODEL}"
echo "  AGENT           : ${AGENT}"
echo "  EVOLUTION_DIR   : ${EVOLUTION_DIR:-<use default path>}"
echo "  VISUAL_LAYER    : ${ENABLE_VISUAL_LAYER}"
echo "  EVO_MODE_TAG    : ${EVO_MODE_TAG}"
echo "============================================================"

bash "${LV_HARNESS_PATH}/run_m3bench_batch.sh"
