"""
Data type definitions: ClipData, TemporalQuestion, AgentAnswer, RetrievalResult, etc.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class ClipData:
    """Data for a single clip."""
    clip_id: int
    base64_video: str                   # base64-encoded video (with audio)
    base64_frames: List[str]            # list of base64-encoded video frames
    base64_audio: Optional[str] = None  # base64-encoded audio (optional)
    path: str = ""                      # clip file path


@dataclass
class TemporalQuestion:
    """Question-answer data with timestamps."""
    question_id: str
    question: str
    options: List[str]
    answer: str
    answerable_after_clip: int = -1     # earliest clip_id at which it is answerable; -1 means the last clip
    ask_at_clips: List[int] = field(default_factory=list)  # at which clips to trigger the question
    category: str = ""                  # question category
    difficulty: str = ""                # difficulty
    video_name: str = ""                # name of the owning video
    mem_path: str = ""                  # corresponding VideoGraph path
    before_clip: Optional[int] = None   # cutoff clip (for compatibility with the existing format)


@dataclass
class RetrievalResult:
    """Retrieval result."""
    event_info: Optional[Dict[str, Any]] = None   # EventGraph retrieval result
    memories: Optional[Dict[str, Any]] = None      # VideoGraph retrieval result
    source: str = ""                               # retrieval source identifier
    raw_payload: Optional[Dict[str, Any]] = None   # raw retrieval payload


@dataclass
class AgentAnswer:
    """The agent's answer result."""
    content: str                        # answer text
    confidence: float = 1.0             # confidence [0, 1]
    is_final: bool = True               # whether this is the final answer
    reasoning_trace: List[Dict] = field(default_factory=list)  # reasoning process record
    search_queries: List[str] = field(default_factory=list)    # retrieval queries used
    num_rounds: int = 0                 # number of reasoning rounds
    tokens_used: int = 0                # token consumption
    conversations: List[Dict] = field(default_factory=list)    # full conversation history


@dataclass
class EvalRecord:
    """Evaluation record for a single question-answer."""
    question_id: str
    clip_id: int
    answer: str
    ground_truth: str
    is_correct: bool
    confidence: float = 1.0
    num_rounds: int = 0
    tokens_used: int = 0
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    question: str = ""
    category: str = ""
    video_name: str = ""
    search_queries: List[str] = field(default_factory=list)


@dataclass
class MemorySnapshot:
    """Snapshot of the memory state."""
    clip_id: int
    data: Any = None                    # serialized memory data
    stats: Dict[str, Any] = field(default_factory=dict)
