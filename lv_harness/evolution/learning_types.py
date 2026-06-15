"""
Learning data type definitions.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class LearningType(Enum):
    """Experience event types."""
    ERROR_CORRECTION = "error"
    SEARCH_FAILURE = "search_fail"
    SEARCH_SUCCESS = "search_win"
    HARD_WIN = "hard_win"
    STRATEGY_INSIGHT = "strategy"
    MEMORY_GAP = "memory_gap"
    PROMPT_IMPROVEMENT = "prompt"
    CONFIDENCE_CALIBRATION = "calibration"
    # P0: subtask-level experience types (from Ledger snapshot)
    SUBTASK_STALL = "subtask_stall"        # subtask stalled (attempts exhausted but still not resolved)
    DECOMPOSE_WIN = "decompose_win"        # whole question correct after multi-subtask decomposition (effective decomposition pattern)
    DECOMPOSE_FAIL = "decompose_fail"      # whole question incorrect after multi-subtask decomposition (ineffective decomposition pattern)
    DECOMPOSE_MISS = "decompose_miss"      # should have decomposed but Planner kept it atomic and got it wrong


@dataclass
class Learning:
    """A single experience record."""
    learning_id: str
    learning_type: LearningType
    video_name: str
    question_id: str
    clip_id: int

    # Context
    question: str
    agent_answer: str
    ground_truth: str
    is_correct: bool
    confidence: float

    # Question type (used for SkillPromoter composite-key clustering)
    # Prefer TemporalQuestion.category; when empty, infer using keyword rules.
    # P2: question_type denotes the "primary tag" (used as the clustering key), while
    # question_type_tags keeps all single tags of this question for SkillRouter to
    # perform auxiliary routing matching and statistical analysis.
    question_type: str = "general"
    question_type_tags: List[str] = field(default_factory=list)

    # Experience content
    what_happened: str = ""
    why_it_happened: str = ""
    what_to_do_next: str = ""

    # Subtask-level metadata (P0: from Ledger snapshot)
    subtask_question: str = ""             # subtask text (filled for SUBTASK_STALL / DECOMPOSE_WIN)
    subtask_count: int = 0                 # total number of subtasks for this question
    resolved_count: int = 0                # number of resolved subtasks for this question
    subtask_status: str = ""               # status of the subtask that triggered the experience
    subtask_attempts: int = 0              # number of attempts of the subtask that triggered the experience
    abandoned_count: int = 0               # number of abandoned subtasks for this question
    partial_count: int = 0                 # number of unfinished partial/searching/pending subtasks for this question
    plan_version: int = 0                  # Ledger plan_version, used to detect frequent replan
    ledger_notes: List[str] = field(default_factory=list)
    ledger_stats: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    search_queries_used: List[str] = field(default_factory=list)
    search_strategy_used: str = ""
    num_rounds: int = 0
    timestamp: str = ""

    # Evolution state
    promoted: bool = False
    promotion_target: Optional[str] = None
