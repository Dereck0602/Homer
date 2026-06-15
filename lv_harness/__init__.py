"""
LV-Harness: a Harness architecture framework for streaming long-video reasoning agents.

Core components:
- HarnessOrchestrator: the time-driven core scheduler
- MemoryStrategy: the abstract interface for memory strategies
- ReasoningAgent: the abstract interface for reasoning agents
- StreamingEvaluator: the streaming evaluator
- HookSystem: the lifecycle hook system
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Vendor path registration: add lv_harness/vendor to sys.path so that
# `from mmagent.xxx import ...` resolves automatically to vendor/mmagent/.
# This makes lv_harness self-contained for the mmagent modules, with no external m3-agent dependency.
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys

_vendor_dir = _os.path.join(_os.path.dirname(__file__), "vendor")
if _vendor_dir not in _sys.path:
    _sys.path.insert(0, _vendor_dir)

# ---------------------------------------------------------------------------

from .orchestrator import HarnessOrchestrator
from .hooks import HookSystem
from .config import load_config
