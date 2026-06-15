"""
YAML configuration loading and parsing.
"""
import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "task": "videomme_streaming",
    "description": "Video-MME streaming evaluation task",

    "data": {
        "annotation_file": "data/annotations/videomme.json",
        "clip_dir_template": "data/video_clips/{video_name}",
        "mem_dir": "data/mems",
        "temporal_mode": "end_of_video",
    },

    "memory": {
        "strategy": "hierarchical",
        "memory_config_path": "configs/memory_config.json",
        "eventgraph_dir": "",
        "memory_cache_dir": "",         # memory cache directory in streaming mode (empty means no cache)
        "topk": 2,
        "mem_wise_topk": 10,           # topk for mem_wise in VIDEO: drilldown mode
        "videograph": {},
        "snapshot": {
            "enabled": True,
            "interval": 10,
            "max_snapshots": 50,
        },
        # sliding-window strategy parameters
        "window_size": 20,
        # compression strategy parameters
        "recent_window": 20,
        "compress_interval": 10,
        # EventGraph incremental update parameters (streaming mode only)
        "eventgraph_incremental": False,
        "eventgraph_update_interval": 20,
        "eventgraph_model": "DeepSeek-V3.1T",
        # interval for periodically refreshing equivalence relations (refresh face cluster merging every N clips)
        "equivalence_refresh_interval": 5,
        # clip memory generation config (used in online mode)
        "ingestion": {
            "use_api": True,
            "api_model": "gemini-2.5-flash",
            "embedding_model": "text-embedding-v4",
            "api_timeout": 120,
            "max_retries": 3,
            "logging_level": "INFO",
            "processing_config_path": "",  # compatibility mode: points to m3-agent's configs/processing_config.json
        },
    },

    "reasoning": {
        "agent": "ledger_multi_round",
        "backend": "openai",
        "model": "gemini-2.5-flash",
        "max_rounds": 5,
        "temperature": 0.4,
        "max_tokens": 8192,
        "seed": 42,
        "api_config_path": "configs/api_config.json",
        "workers": 4,
        "answer_policy": "always",
        # Harness Engineering: constraint and verification config
        "guardrails": {
            "max_tokens_per_question": 50000,
            "max_time_per_question": 300.0,
            "max_retries_on_format_error": 2,
            "max_consecutive_empty_searches": 2,
            "max_similar_queries": 2,
        },
        # Harness Engineering: information sufficiency signal config
        "sufficiency": {
            "enabled": True,
            "increment": {
                "low_increment_threshold": 0.15,   # a new-information ratio below this is considered low increment
                "saturation_rounds": 2,             # saturation is triggered once consecutive low-increment rounds reach this value
            },
            "coverage": {
                "discrimination_threshold": 0.3,    # option coverage difference threshold
            },
        },
        # Harness Engineering: context engineering config
        "context_engineering": {
            "max_conversation_messages": 14,        # max number of messages in conversation history
            "compress_search_results": True,         # whether to compress early retrieval results
        },
    },

    "evaluation": {
        "metrics": ["accuracy"],
        "eval_model": "gemini-2.5-flash",
        "string_match_first": True,
    },

    "output": {
        "dir": "data/results",
        "format": "jsonl",
        "save_intermediate": True,
        "save_conversations": True,
    },

    "evolution": {
        "enabled": False,
        "readonly": False,  # read-only mode: load existing skills for reasoning injection, but write no learning/skill/wisdom
        "dir": ".lv_harness",
        "learnings_dir": ".lv_harness/learnings",
        "skills_dir": ".lv_harness/skills",
        "wisdom_path": ".lv_harness/WISDOM.md",
        "reflections_dir": ".lv_harness/reflections",
        "capture_successes": True,
        "promote_threshold": 3,
        "route_threshold": 0.3,
        "router_min_downstream_samples": 5,
        "router_disable_margin": 0.05,
        "inject_wisdom": True,
        # cross-run reuse switch: when True, startup loads historical skill md from skills_dir
        "load_prior_skills": False,
        # WisdomDistiller upgrade: when True, call the LLM to produce strategy-level insights
        "wisdom_use_llm": False,
        "wisdom_llm_model": "gemini-2.5-flash",
        "wisdom_llm_max_failures": 20,
        "reflection_llm_max_tokens": 8192,
        # P1: SkillPromoter generates special_instructions via the LLM
        "skill_use_llm_instructions": False,
        "skill_instructions_llm_model": "gemini-2.5-flash",
        # whether the skill generation protocol allows the KEYFRAME action. Off by default; set to True together when the script enables the visual layer.
        "visual_layer_enabled": False,
    },
}


def load_config(config_path: str = None, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load a YAML config file, merging the default config and command-line overrides.

    Args:
        config_path: path to the YAML config file
        overrides: command-line override parameters (e.g. {"memory.strategy": "sliding_window"})

    Returns:
        the merged config dict
    """
    config = _deep_copy(DEFAULT_CONFIG)

    # Load the YAML file
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, yaml_config)
        logger.info(f"Loaded config file: {config_path}")

    # Apply command-line overrides
    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key, value)

    return config


def _deep_copy(d):
    """Deep-copy a dict."""
    import copy
    return copy.deepcopy(d)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict, key: str, value):
    """Set a value in a nested dict. key format like "memory.strategy"."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
