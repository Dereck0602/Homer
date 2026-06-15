# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import openai
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.CRITICAL)
# Disable urllib3 logging (which httpx uses)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
# Disable httpcore logging (which httpx uses)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

# api utils

from mmagent._config_loader import get_processing_config, get_api_config
processing_config = get_processing_config()
temp = processing_config["temperature"]

def _load_client(model: str) -> openai.OpenAI:
    """Create and cache an OpenAI-compatible client on demand (see chat_qwen_api.py)."""
    if not hasattr(_load_client, "_cache"):
        _load_client._cache = {}

    cache = _load_client._cache
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

    cache[model] = openai.OpenAI(base_url=base_url, api_key=api_key)
    return cache[model]


def _load_config():
    """Load the api_config.json config (used to get parameters such as qpm)."""
    if not hasattr(_load_config, "_cache"):
        _load_config._cache = get_api_config()
    return _load_config._cache

MAX_RETRIES = 5

def get_response(model, messages, timeout=30):
    """Get chat completion response from specified model.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
    """
    client = _load_client(model)
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temp, timeout=timeout, max_tokens=8192
    )
    content = response.choices[0].message.content or ""
    total_tokens = getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0
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
    #exit()
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_response(model, messages, timeout=30):
    """Process multiple messages in parallel using ThreadPoolExecutor.
    Messages are processed in batches, with each batch completing before starting the next.

    Args:
        model (str): Model identifier
        messages (list): List of message lists to process

    Returns:
        tuple: (list of responses, total tokens used)
    """
    config = _load_config()
    batch_size = config.get(model, {}).get("qpm", 6)
    responses = []
    total_tokens = 0

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            batch_responses = list(executor.map(lambda msg: get_response_with_retry(model, msg, timeout), batch))
            
        # Extract answers and tokens from batch responses
        batch_answers = [response[0] for response in batch_responses]
        batch_tokens = [response[1] for response in batch_responses]
        
        responses.extend(batch_answers)
        total_tokens += sum(batch_tokens)

    return responses, total_tokens


# embedding dimension cache: used to generate zero-vector placeholders, avoiding index misalignment upstream
_EMBED_DIM_CACHE = {}


def _normalize_embedding_text(text):
    """
    Normalize an arbitrary input into a string suitable for embedding.
    Returns (normalized_str, is_empty).
    - None / empty string / pure whitespace -> is_empty=True
    - dict / list and other structured objects -> try JSON serialization; if non-whitespace after serialization, treat as valid
    - other str()-able objects -> convert to str and check emptiness
    """
    if text is None:
        return "", True
    if isinstance(text, str):
        return text, not text.strip()
    # dict / list and other structured objects
    if isinstance(text, (dict, list, tuple)):
        try:
            s = json.dumps(text, ensure_ascii=False)
        except Exception:
            s = str(text)
        return s, not s.strip() or s.strip() in ("{}", "[]", "()", "null")
    try:
        s = str(text)
    except Exception:
        return "", True
    return s, not s.strip()


def _get_zero_vector(model):
    """Return a zero vector based on the cached dimension, or None if the cache is missing."""
    dim = _EMBED_DIM_CACHE.get(model, 0)
    return [0.0] * dim if dim > 0 else None


def get_embedding(model, text, timeout=15):
    """Get embedding for text using specified model.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
    """
    # Empty-input guard: raise ValueError directly to avoid wasting an API call (the server would return 400)
    norm_text, is_empty = _normalize_embedding_text(text)
    if is_empty:
        raise ValueError(f"empty embedding input for model={model}")
    client = _load_client(model)
    response = client.embeddings.create(input=norm_text, model=model, timeout=timeout)
    emb = response.data[0].embedding
    # Record the embedding dimension, to ease generating placeholder zero vectors in batch calls
    try:
        _EMBED_DIM_CACHE[model] = len(emb)
    except Exception:
        pass
    return emb, response.usage.total_tokens


def get_embedding_with_retry(model, text, timeout=15):
    """Retry get_embedding up to MAX_RETRIES times with error handling.

    Empty input / 400 BadRequest -> return a zero-vector placeholder (instead of raising),
    to avoid a single bad item crashing the whole ``executor.map`` batch and blocking the upstream pipeline.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
    """
    # Local pre-check: empty input returns a placeholder zero vector directly (no request, no exception)
    _, is_empty = _normalize_embedding_text(text)
    if is_empty:
        logger.warning(f"empty embedding input for model={model}, return zero vector")
        return _get_zero_vector(model), 0

    for i in range(MAX_RETRIES):
        try:
            return get_embedding(model, text, timeout)
        except openai.BadRequestError as e:
            # 400 is a parameter/content validation error; retrying is pointless. Return a zero-vector placeholder without breaking the pipeline
            logger.warning(
                f"BadRequest (no retry, return zero vector): {e} "
                f"| text preview: {str(text)[:120]!r}"
            )
            return _get_zero_vector(model), 0
        except ValueError as e:
            # local validation errors such as empty input; also degrade to a zero vector
            logger.warning(f"ValueError (return zero vector): {e}")
            return _get_zero_vector(model), 0
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from get embedding")
            continue
    # All retries failed; degrade to a zero vector instead of raising, to keep the pipeline running
    logger.error(
        f"Failed to get embedding after {MAX_RETRIES} retries, return zero vector "
        f"| text preview: {str(text)[:120]!r}"
    )
    return _get_zero_vector(model), 0


def parallel_get_embedding(model, texts, timeout=15):
    """Process multiple texts in parallel to get embeddings.

    For empty strings / None / non-embeddable content in the input, a zero vector is placed in the
    return value (with dimension aligned to other valid embeddings of this model), to keep the result list
    in one-to-one correspondence with the input list and avoid index misalignment in upstream logic such as
    ``zip(texts, embeddings)``.

    Args:
        model (str): Model identifier
        texts (list): List of texts to embed

    Returns:
        tuple: (list of embeddings, total tokens used)
    """
    config = _load_config()
    batch_size = config.get(model, {}).get("qpm", 6)

    # Count empty inputs for a log hint (the actual filtering is done inside get_embedding_with_retry)
    empty_cnt = sum(1 for t in texts if _normalize_embedding_text(t)[1])
    if empty_cnt > 0:
        logger.warning(
            f"parallel_get_embedding: {empty_cnt}/{len(texts)} empty inputs for model={model}, "
            f"will be filled with zero vectors"
        )

    embeddings = []
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_workers = max(1, len(batch))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda x: get_embedding_with_retry(model, x, timeout), batch))

        batch_embeddings = [result[0] for result in results]
        batch_tokens = [result[1] for result in results]

        embeddings.extend(batch_embeddings)
        total_tokens += sum(batch_tokens)

    return embeddings, total_tokens

def get_whisper(model, file_path):
    """Transcribe audio file using Whisper model.

    Args:
        model (str): Model identifier
        file_path (str): Path to audio file

    Returns:
        str: Transcription text
    """
    client = _load_client(model)
    file = open(file_path, "rb")
    return client.audio.transcriptions.create(model=model, file=file).text

def get_whisper_with_retry(model, file_path):
    """Retry Whisper transcription with error handling.

    Args:
        model (str): Model identifier
        file_path (str): Path to audio file

    Returns:
        str: Transcription text
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_whisper(model, file_path)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e}")
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_whisper(model, file_paths):
    """Process multiple audio files in parallel using Whisper model.

    Args:
        model (str): Model identifier
        file_paths (list): List of audio file paths

    Returns:
        list: List of transcription results
    """
    config = _load_config()
    batch_size = config.get(model, {}).get("qpm", 6)
    responses = []
    
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        max_workers = len(batch)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_responses = list(executor.map(lambda x: get_whisper_with_retry(model, x), batch))
            
        responses.extend(batch_responses)
        
    return responses

def generate_messages(inputs):
    """Generate message list for chat completion from mixed inputs.

    Args:
        inputs (list): List of input dictionaries with 'type' and 'content' keys
        type can be:
            "text" - text content
            "image/jpeg", "image/png" - base64 encoded images
            "video/mp4", "video/webm" - base64 encoded videos
            "video_url" - video URL
            "audio/mp3", "audio/wav" - base64 encoded audio
        content should be a string for text,
        a list of base64 encoded media for images/video/audio,
        or a string (url) for video_url
        inputs are like: 
        [
            {
                "type": "video_base64/mp4",
                "content": <base64>
            },
            {
                "type": "text",
                "content": "Describe the video content."
            },
            ...
        ]

    Returns:
        list: Formatted messages for chat completion
    """
    messages = []
    messages.append(
        {"role": "system", "content": "You are an expert in video understanding."}
    )
    content = []
    for input in inputs:
        if not input["content"]:
            logger.warning("empty content, skip")
            continue
        if input["type"] == "text":
            content.append({"type": "text", "text": input["content"]})
        elif input["type"] in ["images/jpeg", "images/png"]:
            img_format = input["type"].split("/")[1]
            if isinstance(input["content"][0], str):
                content.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{img}",
                                "detail": "high",
                            },
                        }
                        for img in input["content"]
                    ]
                )
            else:
                for img in input["content"]:
                    content.append({
                        "type": "text",
                        "text": img[0],
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_format};base64,{img[1]}",
                            "detail": "high",
                        },
                    })
        elif input["type"] == "video_url":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": input["content"]},
                }
            )
        elif input["type"] in ["video_base64/mp4", "video_base64/webm"]:
            video_format = input["type"].split("/")[1]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:video/{video_format};base64,{input['content']}"},
                }
            )
        elif input["type"] in ["audio_base64/mp3", "audio_base64/wav"]:
            audio_format = input["type"].split("/")[1]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:audio/{audio_format};base64,{input['content']}"
                    },
                }
            )
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages

def print_messages(messages):
    for message in messages:
        if message["role"] == "user":
            for item in message["content"]:
                if item["type"] == "text":
                    logger.debug(item['text'])
