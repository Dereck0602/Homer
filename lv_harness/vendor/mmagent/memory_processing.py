# -*- coding: utf-8 -*-
"""
vendor/mmagent/memory_processing.py: trimmed version.

Contains only the parse_video_caption function referenced by modules such as videograph.py and retrieve.py.
The full memory-processing logic lives in memory_processing_simplified.py and
memory_processing_qwen_simplified.py.
"""
import re
import logging

logger = logging.getLogger(__name__)


def parse_video_caption(video_graph, video_caption):
    """Extract entity references (e.g. <face_1>, <voice_2>) from a video caption."""
    def verify_entity(video_graph, entity_str):
        try:
            node_type, node_id = entity_str.split("_")
            node_type = node_type.strip().lower()
            assert node_type in ["face", "voice", "character"]
            node_id = int(node_id)
            try:
                if entity_str in video_graph.reverse_character_mappings.keys() or entity_str in video_graph.character_mappings.keys():
                    return (node_type, node_id)
            except Exception as e:
                pass
            if (node_type == 'face' and node_id in video_graph.nodes and video_graph.nodes[node_id].type == 'img') or (node_type == 'voice' and node_id in video_graph.nodes and video_graph.nodes[node_id].type == 'voice'):
                return (node_type, node_id)
            return None
        except Exception as e:
            logger.error(f"Entities parsing error: {e}")
            return None

    pattern = r'<([^<>]*_[^<>]*)>'
    entity_strs = re.findall(pattern, video_caption)
    entities = [verify_entity(video_graph, entity_str) for entity_str in entity_strs]
    entities = [entity for entity in entities if entity is not None]
    return entities
