# LV-Harness

**LV-Harness** is a Harness-architecture framework for **streaming long-video reasoning agents**. It drives a time-ordered timeline (per video, then per clip), ingests clip-level memory while "watching", and schedules multi-round retrieval-augmented question answering on top of a hierarchical memory. The framework bundles deterministic guardrails, information-sufficiency signals, context engineering, and an optional self-evolution loop (experience capture -> skill distillation -> skill injection -> wisdom distillation).

---

## Features

- **Time-driven orchestrator** (`HarnessOrchestrator`): a two-level loop over videos and clips, coordinating memory ingestion, QA scheduling, and evaluation.
- **Pluggable memory strategies**: `hierarchical`, `videograph_only`, `eventgraph_only`, `no_graph_walk`, `sliding_window`, `compressed`.
- **Pluggable reasoning agents**: `ledger_multi_round` (default), `multi_round_search`, `decompose_only`, `control_api_harness`.
- **Harness Engineering**: deterministic verification hooks, per-question guardrails (token/time/retry budgets), information-sufficiency saturation detection, and conversation-history compression.
- **Self-evolution (optional)**: captures learnings, promotes recurring patterns into reusable skills, routes skills back into reasoning, and distills run-level wisdom.
- **Streaming and batch modes**: answer at end-of-video or build memory and answer incrementally while streaming.
- **Self-contained `mmagent` vendor**: the required `mmagent` modules are vendored under `lv_harness/vendor/`, so no external m3-agent installation is needed.

## Repository layout

```
lv_harness_anon/
├── lv_harness/                  # core framework package
│   ├── orchestrator.py          # HarnessOrchestrator: the time-driven scheduler
│   ├── config.py                # default config + YAML loading/merging
│   ├── cli.py                   # `python -m lv_harness run ...` entry point
│   ├── hooks.py                 # lifecycle hook system
│   ├── memory/                  # memory strategies (hierarchical, sliding_window, ...)
│   ├── reasoning/               # reasoning agents, task ledger, planner, verification hooks
│   ├── evolution/               # self-evolution: learning capture, skill promoter/router, wisdom distiller
│   ├── evaluation/              # streaming evaluator and metrics
│   ├── tools/                   # auxiliary tools
│   ├── data/                    # dataset types and loaders
│   └── vendor/mmagent/          # vendored mmagent (retrieve, eventgraph, memory_processing, ...)
├── configs/
│   ├── api_config.json          # per-model base_url / api_key (fill in placeholders)
│   ├── memory_config.json       # memory pipeline config
│   ├── processing_config.json   # clip ingestion / processing config
│   └── tasks/                   # task YAMLs (e.g. videomme_streaming.yaml)
├── scripts/                     # analysis helpers (accuracy by type, ledger vs baseline)
├── run_homer.py                 # subset evaluation entry (single-pass)
├── run_homer_batch_evo.py       # batch-level self-evolution evaluation entry
├── run_m3bench_*.sh             # M3-Bench launchers
├── run_videomme_*.sh            # Video-MME launchers
├── run_lvomnibench_*.sh         # LVOmniBench launchers
├── create_keyframe.py           # keyframe extraction for clips (hybrid quality + shot coverage)
├── create_high.py               # build event-level high-level graphs from memory graphs
├── run_create_high.sh           # launcher for create_high.py
└── setup.py
```

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

### Via the CLI

```bash
# Run a task from a YAML config
python -m lv_harness run --config configs/tasks/videomme_streaming.yaml

# Override parameters on the command line
python -m lv_harness run \
    --config configs/tasks/videomme_streaming.yaml \
    --model gemini-2.5-flash \
    --strategy hierarchical \
    --workers 4

# Streaming mode (build memory and answer while watching the video)
python -m lv_harness run --config configs/tasks/videomme_streaming.yaml --streaming
```

### Via the benchmark launchers

The `run_*.sh` scripts are environment-variable driven wrappers around `run_homer.py` / `run_homer_batch_evo.py`. Set `LV_HARNESS_PATH` and the dataset roots at the top of each script, then:

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

## Notes

- This is an anonymized release. All credentials (`base_url`, `api_key`) and dataset paths are replaced with placeholders (`YOUR_BASE_URL`, `YOUR_API_KEY`, `/path/to/...`); fill them in for your environment.
- The vendored `mmagent` modules retain their original Apache-2.0 license headers.
