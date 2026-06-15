from .base import ReasoningAgent
from .multi_round import MultiRoundSearchAgent
from .answer_policy import AnswerPolicy, AlwaysAnswer, ConfidentAnswer, DeferredAnswer

# TaskLedger-driven agent (an opt-in extension that is non-invasive to multi_round).
# Active only when the user enables `reasoning.agent: "ledger_multi_round"` via config.
try:
    from .ledger_agent import LedgerAwareMultiRoundAgent  # noqa: F401
    from .task_ledger import TaskLedger, SubTask, Evidence, LedgerUpdater  # noqa: F401
    from .planner import QuestionPlanner, PlannerConfig  # noqa: F401
except Exception as _ledger_import_exc:  # fallback: a new module import failure must not break the legacy path
    import logging as _logging
    _logging.getLogger(__name__).debug(
        f"[reasoning.__init__] ledger module not ready, keeping baseline agent: {_ledger_import_exc}"
    )

# DecomposeOnly agent: only performs task decomposition, without injecting any ledger content into the context.
# Active only when the user enables `reasoning.agent: "decompose_only"` via config.
try:
    from .decompose_only_agent import DecomposeOnlyMultiRoundAgent  # noqa: F401
except Exception as _do_import_exc:
    import logging as _logging
    _logging.getLogger(__name__).debug(
        f"[reasoning.__init__] decompose_only module not ready, keeping other agents: {_do_import_exc}"
    )

# ControlApiHarness agent: wraps the control_api.py reasoning logic as an lv_harness Agent,
# keeping the original prompt and retrieval unchanged, while adding the harness machinery (guardrails / sufficiency / evolution).
try:
    from .control_api_harness_agent import ControlApiHarnessAgent  # noqa: F401
except Exception as _cah_import_exc:
    import logging as _logging
    _logging.getLogger(__name__).debug(
        f"[reasoning.__init__] control_api_harness module not ready: {_cah_import_exc}"
    )
