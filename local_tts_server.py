"""Single GPU owner for all local voice-bot TTS. Bind to loopback, one worker."""
import gc
import logging
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Local GPU TTS")
_lock = threading.Lock()
_model = None
_loaded = None


class Selection(BaseModel):
    engine: Literal["kokoro", "chatterbox", "cloud"]


class Speech(BaseModel):
    engine: Literal["kokoro", "chatterbox"] = "chatterbox"
    text: str = Field(min_length=1, max_length=2000)
    voice: str = "af_heart"
    lang_code: str = "a"
    repo_id: str = "hexgrad/Kokoro-82M"
    exaggeration: float = Field(default=0.7, ge=0, le=1)
    cfg_weight: float = Field(default=0.3, ge=0, le=1)


def _unload():
    global _model, _loaded
    _model = None
    _loaded = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(req):
    global _model, _loaded
    key = (req.engine, req.repo_id, req.lang_code) if req.engine == "kokoro" else (req.engine,)
    if _loaded == key:
        return _model
    _unload()
    if not torch.cuda.is_available():
        raise RuntimeError("Local TTS requires an NVIDIA GPU with CUDA; CPU fallback is disabled.")
    if req.engine == "chatterbox":
        from chatterbox.tts import ChatterboxTTS
        _model = ChatterboxTTS.from_local(ROOT / "models" / "chatterbox", device="cuda")
        _model.watermarker.perth_net.to("cuda")
    else:
        from kokoro import KPipeline
        _model = KPipeline(lang_code=req.lang_code, repo_id=req.repo_id, device="cuda")
    _loaded = key
    return _model


@app.get("/health")
def health():
    return {"service": "local-tts", "engine": _loaded[0] if _loaded else None,
            "loaded_models": 1 if _loaded else 0, "busy": _lock.locked(),
            "device": "cuda", "cuda_available": torch.cuda.is_available(),
            "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1)}


@app.post("/select")
def select(req: Selection):
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "TTS is generating audio. Wait for it to finish before switching.")
    try:
        if _loaded and _loaded[0] != req.engine:
            _unload()
        return {"selected": req.engine, "loaded_models": 1 if _loaded else 0}
    finally:
        _lock.release()


@app.post("/unload")
def unload():
    return select(Selection(engine="cloud"))


@app.post("/tts")
def speech(req: Speech):
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "TTS is already generating audio. Wait before testing again.")
    started = time.monotonic()
    try:
        with torch.inference_mode():
            model = _load(req)
            if req.engine == "chatterbox":
                audio = model.generate(req.text, exaggeration=req.exaggeration,
                                       cfg_weight=req.cfg_weight).detach().cpu().numpy().reshape(-1)
                if model.sr != 24000:
                    raise RuntimeError("Unexpected Chatterbox sample rate")
            else:
                chunks = [a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
                          for _, _, a in model(req.text, voice=req.voice) if a is not None]
                audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
            if not audio.size:
                raise RuntimeError("TTS generated no audio")
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
            return Response(pcm, media_type="application/octet-stream",
                            headers={"X-Sample-Rate": "24000",
                                     "X-Generation-Seconds": str(round(time.monotonic() - started, 2))})
    except (Exception, SystemExit) as exc:
        logging.getLogger(__name__).exception("Local TTS generation failed")
        message = str(exc)
        model = None
        _unload()
        raise HTTPException(503, message) from None
    finally:
        _lock.release()
