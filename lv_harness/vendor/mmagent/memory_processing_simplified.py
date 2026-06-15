# -*- coding: utf-8 -*-
"""
Plan D API edition: simplified memory processing module (compatible with Gemini / OpenAI-compatible APIs).

Differences from memory_processing_qwen_simplified.py:
  - Uses the OpenAI-compatible API calls in chat_api.py, replacing the local Qwen inference in chat_qwen.py
  - Video input changes from file paths to base64 encoding (passed to the API via a data URI)
  - Supports specifying the model through the "api_model" field in processing_config.json (default gemini-2.5-flash)
  - Gemini natively supports audio understanding within video, so it can directly extract speech content from the video

Core changes:
  - Removes the entire voiceprint processing pipeline (M1 qwen3-omni ASR + M2 ERes2NetV2 + voiceprint clustering)
  - Keeps only face processing (M3 InsightFace)
  - The API model (e.g., Gemini) directly understands speech content from video plus audio and attributes it to face_id
  - Uses the text-embedding-v4 API for text embedding

Model list (3 models):
  M3: InsightFace/buffalo_l (local CPU, face detection + embedding + clustering)
  M4: Gemini 2.5 Flash or another OpenAI-compatible model (API call, multimodal understanding + ASR + memory generation)
  M5: text-embedding-v4 (API call, text embedding)
"""
import base64
import json
import logging
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .utils.chat_api import generate_messages, get_response_with_retry
from .utils.chat_api import parallel_get_embedding
from .utils.general import validate_and_fix_json
from .prompts_simplified import (
    prompt_generate_memory_simplified,
    prompt_generate_memory_simplified_no_face,
)
from .memory_processing import parse_video_caption

from mmagent._config_loader import get_processing_config; processing_config = get_processing_config()
logging_level = processing_config["logging"]
MAX_RETRIES = processing_config["max_retries"]

# API model name, read from processing_config, defaults to gemini-2.5-flash
API_MODEL = processing_config.get("api_model", "gemini-2.5-flash")

# Embedding model used by Plan D, preferring the configured text-embedding-v4
EMBEDDING_MODEL = processing_config.get("embedding_model", "text-embedding-v4")

# API call timeout (seconds); video understanding tasks usually require a longer time
API_TIMEOUT = processing_config.get("api_timeout", 120)

logger = logging.getLogger(__name__)


def generate_video_context_simplified(
    base64_frames, faces_list, base64_video=None, faces_input="face_only"
):
    """
    Generate the simplified video context (no voiceprint information, using base64-encoded video).

    Differences from the Qwen version:
      - Video is passed via base64 encoding (rather than file path), adapted for the OpenAI-compatible API
      - The API model (e.g., Gemini) automatically extracts audio from the video and performs ASR

    Args:
        base64_frames: list of base64-encoded video frames
        faces_list: mapping from face_id to a list of faces
        base64_video: base64-encoded full video (with audio track)
        faces_input: face input type, "face_only" or "face_frames"

    Returns:
        list: list of video context items for the API model to process
    """
    face_frames = []
    face_only = []

    for char_id, faces in faces_list.items():
        if len(faces) == 0:
            continue
        face = faces[0]
        frame_id = face["frame_id"]
        frame_base64 = base64_frames[frame_id]

        # Draw the face bounding box
        frame_bytes = base64.b64decode(frame_base64)
        frame_img = Image.open(BytesIO(frame_bytes))
        draw = ImageDraw.Draw(frame_img)
        bbox = face["bounding_box"]
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=(0, 255, 0), width=4
        )
        buffered = BytesIO()
        frame_img.save(buffered, format="JPEG")
        frame_base64 = base64.b64encode(buffered.getvalue()).decode()
        face_frames.append((f"<face_{char_id}>:", frame_base64))
        face_only.append((f"<face_{char_id}>:", face["extra_data"]["face_base64"]))

    if faces_input == "face_only":
        faces_input_data = face_only
    elif faces_input == "face_frames":
        faces_input_data = face_frames
    else:
        raise ValueError(f"Invalid face input: {faces_input}")

    num_faces = len(faces_input_data)
    if num_faces == 0:
        logger.warning("No qualified faces detected")

    # Visualization logging
    if logging_level == "DETAIL" and num_faces > 0:
        num_rows = (num_faces + 2) // 3
        _, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.ravel()
        for i, face_pic in enumerate(faces_input_data):
            img_bytes = base64.b64decode(face_pic[1])
            img_array = np.array(Image.open(BytesIO(img_bytes)))
            axes[i].imshow(img_array)
            axes[i].set_title(face_pic[0])
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

    # Build the video context: use the base64-encoded video (with audio track)
    # The generate_messages function in chat_api.py converts it to the data:video/mp4;base64,... format
    video_context = []
    if base64_video:
        # Ensure base64_video is a string
        if isinstance(base64_video, bytes):
            base64_video = base64_video.decode("utf-8")
        video_context.append(
            {"type": "video_base64/mp4", "content": base64_video}
        )
    else:
        # If there is no base64 video, fall back to using frame images
        logger.warning("No base64_video provided, falling back to frame images")
        video_context.append(
            {"type": "images/jpeg", "content": [f for f in base64_frames]}
        )

    video_context.append({"type": "text", "content": "Face features:"})
    if faces_input_data:
        video_context.append({"type": "images/jpeg", "content": faces_input_data})

    return video_context


def generate_all_memories_simplified(video_context, has_faces=True):
    """
    Use an API model (e.g., Gemini) to generate memories (simplified, no voiceprint).

    The API model performs ASR on its own through the video's audio channel
    and attributes the speech to the corresponding face_id.

    Args:
        video_context: list of video context items
        has_faces: whether face features are included

    Returns:
        tuple: (episodic_memories, semantic_memories)
    """
    # Choose a different prompt depending on whether faces are present
    if has_faces:
        prompt = prompt_generate_memory_simplified
    else:
        prompt = prompt_generate_memory_simplified_no_face

    input_data = [
        {"type": "text", "content": prompt},
    ] + video_context

    messages = generate_messages(input_data)

    epi_key = "episodic_memory"
    sem_key = "semantic_memory"

    memories = None
    last_raw_response = ""
    for i in range(MAX_RETRIES):
        try:
            memories_string = get_response_with_retry(
                API_MODEL, messages, timeout=API_TIMEOUT
            )[0]
        except Exception as e:
            logger.error(f"API call failed on attempt {i+1}/{MAX_RETRIES}: {e}")
            memories_string = ""

        last_raw_response = memories_string or ""

        # Empty response -> treat as failure and keep retrying instead of falling back to "[]"
        if not memories_string:
            logger.warning(
                f"Empty response from memory model on attempt {i+1}/{MAX_RETRIES}, retrying..."
            )
            continue

        parsed = validate_and_fix_json(memories_string)

        # Only a parsed dict counts as success; list / None / other types are treated as failure
        if isinstance(parsed, dict):
            memories = parsed
            break
        else:
            logger.warning(
                f"Memory model returned non-dict JSON on attempt {i+1}/{MAX_RETRIES} "
                f"(type={type(parsed).__name__}), raw[:200]={memories_string[:200]!r}"
            )

    if not isinstance(memories, dict):
        logger.error(
            f"Failed to obtain valid memory dict after {MAX_RETRIES} attempts. "
            f"Falling back to empty memories. Last raw response[:200]={last_raw_response[:200]!r}"
        )
        memories = {epi_key: [], sem_key: []}

    episodic_memories = memories.get(epi_key, []) or []
    semantic_memories = memories.get(sem_key, []) or []

    # Defensive: downstream expects a list; if the model returns an abnormal structure such as dict/str, degrade to empty
    if not isinstance(episodic_memories, list):
        logger.warning(
            f"episodic_memories is not a list (got {type(episodic_memories).__name__}), coerced to []"
        )
        episodic_memories = []
    if not isinstance(semantic_memories, list):
        logger.warning(
            f"semantic_memories is not a list (got {type(semantic_memories).__name__}), coerced to []"
        )
        semantic_memories = []

    return episodic_memories, semantic_memories


def generate_memories_simplified(base64_frames, faces_list, base64_video):
    """
    Memory generation entry point for the Plan D API edition.

    Differences from the Qwen version:
      - The third argument changes from video_path (file path) to base64_video (base64-encoded video)
      - The video is passed to the API via base64 encoding, and the API model handles the audio itself

    Args:
        base64_frames: list of base64-encoded video frames
        faces_list: mapping from face_id to a list of faces (from process_faces)
        base64_video: base64-encoded full video (with audio track)

    Returns:
        tuple: (episodic_memories, semantic_memories)
    """
    has_faces = len(faces_list) > 0
    video_context = generate_video_context_simplified(
        base64_frames, faces_list, base64_video
    )
    episodic_memories, semantic_memories = generate_all_memories_simplified(
        video_context, has_faces=has_faces
    )
    return episodic_memories, semantic_memories


def process_memories_simplified(
    video_graph, memory_contents, clip_id, type="episodic"
):
    """
    Process memories and write them into the VideoGraph (simplified, using a configurable embedding model).

    Differences from the original process_memories:
      - Uses a configurable embedding model (default text-embedding-v4)
      - All other logic is identical to the original

    Args:
        video_graph: VideoGraph instance
        memory_contents: list of memory contents (list of strings)
        clip_id: ID of the current clip
        type: memory type, "episodic" or "semantic"
    """

    def get_memory_embeddings(memory_contents):
        """Compute memory vectors using the configured embedding model"""
        embeddings = parallel_get_embedding(EMBEDDING_MODEL, memory_contents)[0]
        return embeddings

    def insert_memory(video_graph, memory, type="episodic"):
        """Create a new text node and establish edges"""
        new_node_id = video_graph.add_text_node(memory, clip_id, type)
        entities = parse_video_caption(video_graph, memory["contents"][0])
        for entity in entities:
            video_graph.add_edge(new_node_id, entity[1])

    def update_video_graph(video_graph, memories, type="episodic"):
        """Update the VideoGraph"""
        if type == "episodic":
            for memory in memories:
                insert_memory(video_graph, memory, type)
        elif type == "semantic":
            for memory in memories:
                entities = parse_video_caption(
                    video_graph, memory["contents"][0]
                )

                if len(entities) == 0:
                    insert_memory(video_graph, memory, type)
                    continue

                positive_threshold = 0.85
                negative_threshold = 0

                node_id = entities[0][1]
                related_nodes = video_graph.get_connected_nodes(
                    node_id, type=["semantic"]
                )

                create_new_node = True
                for rn_id in related_nodes:
                    related_node_entities = parse_video_caption(
                        video_graph,
                        video_graph.nodes[rn_id].metadata["contents"][0],
                    )
                    embedding = video_graph.nodes[rn_id].embeddings[0]
                    if all(
                        entity in related_node_entities
                        for entity in entities
                    ):
                        similarity = np.dot(
                            memory["embeddings"][0], embedding
                        ) / (
                            np.linalg.norm(memory["embeddings"][0])
                            * np.linalg.norm(embedding)
                        )
                        if similarity > positive_threshold:
                            video_graph.reinforce_node(rn_id)
                            create_new_node = False
                        elif similarity < negative_threshold:
                            video_graph.weaken_node(rn_id)
                            create_new_node = False

                if create_new_node:
                    insert_memory(video_graph, memory, type)

    if not memory_contents:
        return

    memories_embeddings = get_memory_embeddings(memory_contents)

    memories = []
    for memory, embedding in zip(memory_contents, memories_embeddings):
        memories.append({"contents": [memory], "embeddings": [embedding]})

    update_video_graph(video_graph, memories, type)
