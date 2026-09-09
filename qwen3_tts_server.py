"""Local Qwen3-TTS voice-cloning service shared by all voice bots.

The service deliberately lives in its own Python environment because Qwen3-TTS
has newer ML dependencies than the existing Kokoro/Whisper application.
"""

from __future__ import annotations

import os
import gc
import threading
from typing import Literal
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
PRESET_MODEL_ID = os.getenv("QWEN3_TTS_PRESET_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
SPEAKERS = ("Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee")
API_KEY = os.getenv("QWEN3_TTS_API_KEY", "")
DEVICE = os.getenv("QWEN3_TTS_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTENTION = os.getenv("QWEN3_TTS_ATTENTION", "sdpa")

app = FastAPI(title="Qwen3-TTS Voice Clone Service")
_model = None
_loaded_model_id = None
_model_lock = threading.RLock()
_prompt_cache: dict[tuple[str, str], object] = {}


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str = "Auto"
    voice_mode: Literal["preset", "clone"] = "preset"
    speaker: str = "Ryan"
    ref_audio: str = ""
    ref_text: str = ""


def _authorize(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid Qwen3-TTS API key")


def _get_model(model_id):
    global _model, _loaded_model_id
    if _model is not None and _loaded_model_id == model_id:
        return _model
    with _model_lock:
        if _model is None or _loaded_model_id != model_id:
            _model = None
            _loaded_model_id = None
            _prompt_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from qwen_tts import Qwen3TTSModel

            dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
            _model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=DEVICE,
                dtype=dtype,
                attn_implementation=ATTENTION,
            )
            _loaded_model_id = model_id
    return _model


def _validate_reference(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme in ("http", "https", "data"):
        return
    if len(value) > 256 and not any(sep in value for sep in ("/", "\\")):
        return  # Qwen also accepts base64 audio strings.
    if not Path(value).expanduser().is_file():
        raise HTTPException(400, f"Reference audio file not found: {value}")


@app.get("/health")
def health():
    return {"status": "ok", "model": _loaded_model_id, "device": DEVICE,
            "loaded": _model is not None, "speakers": SPEAKERS}


@app.post("/tts")
def synthesize(item: TTSRequest, authorization: str | None = Header(default=None)):
    _authorize(authorization)
    if item.voice_mode == "clone":
        if not item.ref_audio.strip():
            raise HTTPException(400, "Reference audio is required for My Voice")
        _validate_reference(item.ref_audio)
    elif item.speaker not in SPEAKERS:
        raise HTTPException(400, "Unknown Qwen3 preset voice")

    with _model_lock:
        model = _get_model(MODEL_ID if item.voice_mode == "clone" else PRESET_MODEL_ID)
        if item.voice_mode == "preset":
            wavs, sample_rate = model.generate_custom_voice(
                text=item.text, language=item.language or "Auto", speaker=item.speaker,
            )
        else:
            cache_key = (item.ref_audio, item.ref_text)
            prompt = _prompt_cache.get(cache_key)
            if prompt is None:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=item.ref_audio, ref_text=item.ref_text or None,
                    x_vector_only_mode=not bool(item.ref_text),
                )
                _prompt_cache.clear()
                _prompt_cache[cache_key] = prompt
            wavs, sample_rate = model.generate_voice_clone(
                text=item.text, language=item.language or "Auto", voice_clone_prompt=prompt,
            )

    if sample_rate != 24000:
        raise HTTPException(500, f"Unexpected Qwen3-TTS sample rate: {sample_rate}")
    audio = np.asarray(wavs[0], dtype=np.float32)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return Response(content=pcm, media_type="application/octet-stream",
                    headers={"X-Audio-Format": "pcm_s16le", "X-Sample-Rate": "24000"})
