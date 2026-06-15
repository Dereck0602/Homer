# LV-Harness

**LV-Harness** is a Harness-architecture framework for **streaming long-video reasoning agents**. It drives a time-ordered timeline (per video, then per clip), ingests clip-level memory while "watching", and schedules multi-round retrieval-augmented question answering on top of a hierarchical memory. The framework bundles deterministic guardrails, information-sufficiency signals, context engineering, and an optional self-evolution loop.

---

## Features

- **Pluggable memory strategies**: `hierarchical`, `videograph_only`, `eventgraph_only`, `no_graph_walk`, `sliding_window`, `compressed`.
- **Pluggable reasoning agents**: `ledger_multi_round` (default), `multi_round_search`, `decompose_only`, `control_api_harness`.
- **Harness Engineering**: deterministic verification hooks, per-question guardrails (token/time/retry budgets), information-sufficiency saturation detection, and conversation-history compression.
- **Self-evolution (optional)**: captures learnings, promotes recurring patterns into reusable skills, routes skills back into reasoning, and distills run-level wisdom.


## Installation

```bash
# Python >= 3.8
pip install -e .

# Optional extras for the OpenAI-compatible backend
pip install -e ".[openai]"
```

Core dependencies: `pyyaml`, `numpy`, `tqdm`. The reasoning/ingestion paths additionally use `openai` (any OpenAI-compatible endpoint). The data-prep scripts use `opencv-python`, `pandas`, and `scikit-learn`.

## Configuration

Model endpoints are defined in `configs/api_config.json`. The released config ships with placeholders that must be filled in before running:

```json
{
  "gemini-2.5-flash": {
    "base_url": "YOUR_BASE_URL",
    "api_key": "YOUR_API_KEY"
  }
}
```

Task behavior is driven by a YAML file merged on top of `DEFAULT_CONFIG` in `lv_harness/config.py`. Any nested key can be overridden from the command line, e.g. `--reasoning.model gemini-2.5-flash`. Key defaults:

- `reasoning.agent`: `ledger_multi_round`
- `memory.strategy`: `hierarchical`
- `evaluation.eval_model`: `gemini-2.5-flash`
- `evolution.enabled`: `false`

## Quick start

```bash
# M3-Bench, batch-level self-evolution
DATASET=robot NUM_QUESTIONS=128 BATCH_SIZE=64 bash run_m3bench_batch.sh

# Streaming mode
MODE=streaming bash run_m3bench_batch.sh

# Ablation: VideoGraph only (no EventGraph / keyframe / hierarchical retrieval)
STRATEGY=videograph_only bash run_m3bench_batch.sh
```

Common environment variables: `DATASET`, `NUM_QUESTIONS`, `BATCH_SIZE`, `MODEL`, `STRATEGY`, `AGENT`, `MODE`, `WORKERS`, `MAX_ROUNDS`, `EVAL_MODEL`.

## Data preparation

```bash
# 1. Extract keyframes from video clips (single-decode hybrid: sharpness + color entropy + shot coverage)
python create_keyframe.py        # edit ROOT_DIR inside the script

# 2. Build event-level high-level graphs from per-video memory graphs (.pkl)
bash run_create_high.sh --folder_path /path/to/memory_graphs --out_dir /path/to/event_graphs --model gemini-3-flash-preview
```

## Memory strategies

| Strategy          | Description                                                              |
|-------------------|--------------------------------------------------------------------------|
| `hierarchical`    | VideoGraph + EventGraph + keyframe + hierarchical retrieval (default)    |
| `no_graph_walk`   | EventGraph node retrieval + VideoGraph, without graph traversal/keyframe |
| `videograph_only` | VideoGraph retrieval only (ablation)                                     |
| `eventgraph_only` | EventGraph retrieval only                                                |
| `sliding_window`  | Recent-window memory                                                     |
| `compressed`      | Periodic compression of older memory                                     |

## Reasoning agents

| Agent                 | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `ledger_multi_round`  | Task-ledger-driven multi-round search with planning (default)               |
| `multi_round_search`  | Baseline multi-round retrieve-then-answer                                   |
| `decompose_only`      | Question decomposition only, no ledger content injected into context        |
| `control_api_harness` | Wraps a control-API reasoning loop with harness guardrails/sufficiency/evolution |

## Self-evolution (optional)

When enabled (`--evolution`), each answered question can capture a learning; recurring learnings are promoted into reusable skills, routed back into the reasoning prompt, and periodically distilled into run-level wisdom. Storage defaults to `.lv_harness/` (`learnings/`, `skills/`, `WISDOM.md`, `reflections/`).

