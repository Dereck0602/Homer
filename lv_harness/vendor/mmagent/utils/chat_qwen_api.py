# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen (OpenAI-compatible) chat helper.

This repo originally uses Gemini for some multimodal calls. If you don't have a
Gemini key, you can point these calls to Qwen Omni models (e.g.
`qwen3-omni-flash`) via an OpenAI-compatible endpoint.

Config:
  - Add an entry in `configs/api_config.json`:
      {
        "qwen3-omni-flash": {"base_url": "...", "api_key": "..."}
      }

The message schema follows OpenAI Chat Completions with multimodal `content`
items (text + image_url/video_url data URIs), which is supported by most
OpenAI-compatible Qwen endpoints.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from time import sleep


logger = logging.getLogger(__name__)

from mmagent._config_loader import get_processing_config, get_api_config
processing_config = get_processing_config()
temp = 0.6 #processing_config.get("temperature", 0.6)
MAX_RETRIES = 5

def _load_client(model: str) -> OpenAI:
    """Create (and cache) an OpenAI client for a given model from api_config."""
    if not hasattr(_load_client, "_cache"):
        _load_client._cache = {}  # type: ignore[attr-defined]

    cache: Dict[str, OpenAI] = _load_client._cache  # type: ignore[attr-defined]
    if model in cache:
        return cache[model]

    cfg = get_api_config()
    if model not in cfg:
        raise KeyError(
            f"Model '{model}' not found in configs/api_config.json. "
            f"Please add it with base_url and api_key."
        )
    base_url = cfg[model].get("base_url")
    api_key = cfg[model].get("api_key")
    if not base_url or not api_key:
        raise ValueError(
            f"configs/api_config.json has empty base_url/api_key for '{model}'."
        )

    cache[model] = OpenAI(base_url=base_url, api_key=api_key)
    return cache[model]

MAX_RETRIES = 5

def get_response(model: str, messages: List[Dict[str, Any]], timeout: int = 30) -> Tuple[str, int]:
    """Call an OpenAI-compatible chat completion endpoint."""
    client = _load_client(model)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        timeout=timeout,
        max_tokens=8192,
    )
    content = resp.choices[0].message.content or ""
    total_tokens = getattr(resp.usage, "total_tokens", 0) if getattr(resp, "usage", None) else 0
    return content, total_tokens

def get_response_with_retry(model, messages, timeout=30):
    """Retry get_response up to MAX_RETRIES times with error handling.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_response(model, messages, timeout)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")


def generate_messages(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate OpenAI-style multimodal messages (text + data URIs)."""
    messages: List[Dict[str, Any]] = []
    messages.append({"role": "system", "content": "You are an expert in video understanding."})

    content: List[Dict[str, Any]] = []
    for item in inputs:
        if not item.get("content"):
            logger.warning("empty content, skip")
            continue
        t = item.get("type")
        if t == "text":
            content.append({"type": "text", "text": item["content"]})
        elif t in ["images/jpeg", "images/png"]:
            img_format = t.split("/")[1]
            if isinstance(item["content"][0], str):
                content.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{img}",
                                "detail": "high",
                            },
                        }
                        for img in item["content"]
                    ]
                )
            else:
                for img in item["content"]:
                    content.append({"type": "text", "text": img[0]})
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{img[1]}",
                                "detail": "high",
                            },
                        }
                    )
        elif t == "video_url":
            content.append({"type": "video_url", "video_url": {"url": item["content"]}})
        elif t in ["video_base64/mp4", "video_base64/webm"]:
            video_format = t.split("/")[1]
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/{video_format};base64,{item['content']}"},
                }
            )
        elif t in ["audio_base64/mp3", "audio_base64/wav"]:
            audio_format = t.split("/")[1]
            # Many OpenAI-compatible providers accept audio as a data URI via "audio_url".
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/{audio_format};base64,{item['content']}"},
                }
            )
        else:
            raise ValueError(f"Invalid input type: {t}")

    messages.append({"role": "user", "content": content})
    return messages


def get_embedding(model, text, timeout=15):
    """Get embedding for text using specified model.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
    """
    client = _load_client(model)
    response = client.embeddings.create(input=text, model=model, timeout=timeout)
    return response.data[0].embedding, response.usage.total_tokens


def get_embedding_with_retry(model, text, timeout=15):
    """Retry get_embedding up to MAX_RETRIES times with error handling.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_embedding(model, text, timeout)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from get embedding")
            continue
    raise Exception(f"Failed to get embedding after {MAX_RETRIES} retries")

def parallel_get_embedding(model, texts, timeout=15):
    """Process multiple texts in parallel to get embeddings.

    Args:
        model (str): Model identifier
        texts (list): List of texts to embed

    Returns:
        tuple: (list of embeddings, total tokens used)
    """
    #config = json.load(open("configs/api_config.json"))
    batch_size = 6
    embeddings = []
    total_tokens = 0
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_workers = len(batch)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda x: get_embedding_with_retry(model, x, timeout), batch))
            
        # Split batch results into embeddings and tokens
        batch_embeddings = [result[0] for result in results]
        batch_tokens = [result[1] for result in results]
        
        embeddings.extend(batch_embeddings)
        total_tokens += sum(batch_tokens)
        
    return embeddings, total_tokens
