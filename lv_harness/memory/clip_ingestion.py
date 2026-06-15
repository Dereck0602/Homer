# -*- coding: utf-8 -*-
"""
ClipIngestion: internalizes the clip memory-generation logic into lv_harness.

Problems solved:
  - Eliminate the direct dependency on m3_agent.memorization_simplified.process_segment_simplified
  - Eliminate the module-level hardcoded json.load("configs/processing_config.json")
  - All configuration is passed in via parameters and managed uniformly by lv_harness's YAML configuration

Core flow (identical to the original):
  1. InsightFace face detection + clustering -> face_id
  2. Multimodal model understanding (video + audio + faces -> memory generation)
  3. text-embedding-v4 text embedding -> write into VideoGraph
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for clip memory generation.

    All parameters can be injected via lv_harness's YAML configuration,
    and no longer depend on m3-agent's configs/processing_config.json.

    Args:
        use_api: whether to use API mode (True=Gemini API, False=Qwen local)
        api_model: the model name in API mode
        embedding_model: the text embedding model name
        api_timeout: the API call timeout (seconds)
        max_retries: the maximum number of retries
        logging_level: the logging level ("DETAIL" displays face visualization)
        intermediate_path: the directory for saving intermediate results
        processing_config_path: the path to the original processing_config.json (compatibility mode)
    """
    use_api: bool = True
    api_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-v4"
    api_timeout: int = 120
    max_retries: int = 3
    logging_level: str = "INFO"
    intermediate_path: str = "/tmp/lv_harness"
    processing_config_path: str = ""

    @classmethod
    def from_harness_config(cls, config: dict) -> "IngestionConfig":
        """Build an IngestionConfig from lv_harness's memory configuration dictionary.

        Prefer the explicit configuration in lv_harness's YAML;
        if processing_config_path is provided, use it as a fallback to load missing fields.
        """
        ingestion_cfg = config.get("ingestion", {})

        # Try to load fallback values from processing_config.json
        fallback = {}
        pcp = ingestion_cfg.get("processing_config_path", "") or config.get("processing_config_path", "")
        if pcp and os.path.exists(pcp):
            try:
                with open(pcp, "r") as f:
                    fallback = json.load(f)
                logger.info(f"[IngestionConfig] loaded fallback configuration from {pcp}")
            except Exception as e:
                logger.warning(f"[IngestionConfig] failed to load {pcp}: {e}")

        return cls(
            use_api=ingestion_cfg.get("use_api", fallback.get("use_api", True)),
            api_model=ingestion_cfg.get("api_model", fallback.get("api_model", "gemini-2.5-flash")),
            embedding_model=ingestion_cfg.get("embedding_model", fallback.get("embedding_model", "text-embedding-v4")),
            api_timeout=ingestion_cfg.get("api_timeout", fallback.get("api_timeout", 120)),
            max_retries=ingestion_cfg.get("max_retries", fallback.get("max_retries", 3)),
            logging_level=ingestion_cfg.get("logging_level", fallback.get("logging", "INFO")),
            intermediate_path=config.get("intermediate_path", "/tmp/lv_harness"),
            processing_config_path=pcp,
        )


# ---- Lazily loaded module references (to avoid crashes on import) ----

_process_faces = None
_generate_memories_api = None
_process_memories_api = None
_generate_memories_qwen = None
_process_memories_qwen = None


def _ensure_face_processing():
    """Lazily load the face-processing module."""
    global _process_faces
    if _process_faces is None:
        from mmagent.face_processing import process_faces
        _process_faces = process_faces
    return _process_faces


def _ensure_memory_processing_api():
    """Lazily load the API-mode memory-processing module."""
    global _generate_memories_api, _process_memories_api
    if _generate_memories_api is None:
        from mmagent.memory_processing_simplified import (
            generate_memories_simplified,
            process_memories_simplified,
        )
        _generate_memories_api = generate_memories_simplified
        _process_memories_api = process_memories_simplified
    return _generate_memories_api, _process_memories_api


def _ensure_memory_processing_qwen():
    """Lazily load the Qwen local-mode memory-processing module."""
    global _generate_memories_qwen, _process_memories_qwen
    if _generate_memories_qwen is None:
        from mmagent.memory_processing_qwen_simplified import (
            generate_memories_simplified,
            process_memories_simplified,
        )
        _generate_memories_qwen = generate_memories_simplified
        _process_memories_qwen = process_memories_simplified
    return _generate_memories_qwen, _process_memories_qwen


def process_clip(
    video_graph,
    base64_video: str,
    base64_frames: list,
    clip_id: int,
    clip_path: str,
    ingestion_config: IngestionConfig,
) -> None:
    """Write the memory of a single clip into VideoGraph.

    This is lv_harness's internalized process_segment_simplified;
    all configuration is passed in via the ingestion_config parameter and does not depend on module-level global variables.

    Flow:
      1. InsightFace face detection + clustering -> face_id
      2. Multimodal model generates memory (video + audio + faces -> episodic/semantic)
      3. text-embedding-v4 embedding -> write into VideoGraph

    Args:
        video_graph: a VideoGraph instance
        base64_video: the base64-encoded video (with audio)
        base64_frames: the list of base64-encoded video frames
        clip_id: clip ID
        clip_path: the clip file path
        ingestion_config: the memory-generation configuration
    """
    save_path = ingestion_config.intermediate_path
    os.makedirs(save_path, exist_ok=True)

    # Step 1: face processing (InsightFace)
    process_faces_fn = _ensure_face_processing()
    try:
        id2faces = process_faces_fn(
            video_graph,
            base64_frames,
            save_path=os.path.join(save_path, f"clip_{clip_id}_faces.json"),
            preprocessing=[],
        )
    except Exception as e:
        logger.warning(f"[ClipIngestion] clip {clip_id} face processing failed: {e}, using empty faces")
        id2faces = {}

    if not id2faces:
        id2faces = {}

    # Step 2: multimodal model generates memory
    if ingestion_config.use_api:
        generate_fn, process_fn = _ensure_memory_processing_api()
        # API mode: pass in the base64-encoded video (with audio track)
        episodic_memories, semantic_memories = generate_fn(
            base64_frames,
            id2faces,
            base64_video,
        )
    else:
        generate_fn, process_fn = _ensure_memory_processing_qwen()
        # Qwen local mode: pass in the video file path
        episodic_memories, semantic_memories = generate_fn(
            base64_frames,
            id2faces,
            clip_path,
        )

    # Step 3: embedding + write into VideoGraph
    process_fn(video_graph, episodic_memories, clip_id, type="episodic")
    process_fn(video_graph, semantic_memories, clip_id, type="semantic")

    logger.debug(
        f"[ClipIngestion] clip {clip_id} complete: "
        f"episodic={len(episodic_memories)}, semantic={len(semantic_memories)}, "
        f"faces={len(id2faces)}"
    )
