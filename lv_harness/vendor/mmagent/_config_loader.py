# -*- coding: utf-8 -*-
"""
vendor/mmagent configuration loader.

Solves the hardcoded module-level `json.load(open("configs/xxx.json"))` problem in the original mmagent.
It locates config files automatically via multi-path probing, supporting the following lookup order:
  1. the directory specified by the environment variable LV_HARNESS_CONFIG_DIR
  2. the lv_harness/configs/ directory (self-contained config)
  3. the configs/ directory under the current working directory (compatible with m3-agent's original behavior)
  4. the m3-agent/configs/ directory (fallback)
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

# Cache for already-loaded configs
_config_cache = {}

# lv_harness project root: vendor/mmagent/_config_loader.py -> up 3 levels
_LV_HARNESS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _find_config_file(filename: str) -> str:
    """Find a config file by priority order."""
    candidates = []

    # 1. directory specified by the environment variable
    env_dir = os.environ.get("LV_HARNESS_CONFIG_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, filename))

    # 2. lv_harness/configs/ directory
    candidates.append(os.path.join(_LV_HARNESS_ROOT, "configs", filename))

    # 3. configs/ under the current working directory
    candidates.append(os.path.join(os.getcwd(), "configs", filename))

    # 4. relative path (compatible with the original behavior)
    candidates.append(os.path.join("configs", filename))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Config file '{filename}' not found; tried the following paths:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def load_config(filename: str) -> dict:
    """Load a config file (with caching).

    Args:
        filename: config file name, e.g. "processing_config.json"

    Returns:
        the config dict
    """
    if filename in _config_cache:
        return _config_cache[filename]

    path = _find_config_file(filename)
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    _config_cache[filename] = config
    logger.debug(f"[mmagent._config_loader] loaded config: {path}")
    return config


def get_processing_config() -> dict:
    """Load processing_config.json."""
    return load_config("processing_config.json")


def get_api_config() -> dict:
    """Load api_config.json."""
    return load_config("api_config.json")


def get_memory_config() -> dict:
    """Load memory_config.json."""
    return load_config("memory_config.json")
