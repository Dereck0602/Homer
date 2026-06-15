# -*- coding: utf-8 -*-
"""
vendor/mmagent: lv_harness's self-contained mmagent module.

This is a trimmed vendored copy of m3-agent/mmagent.
Unlike the original __init__.py, it performs no eager imports here;
all submodules are lazily loaded on demand to avoid triggering unnecessary dependency chains.
"""
import logging

# Silence third-party library logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
logging.getLogger('moviepy.video.io.VideoFileClip').setLevel(logging.ERROR)
logging.getLogger('moviepy.audio.io.AudioFileClip').setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
