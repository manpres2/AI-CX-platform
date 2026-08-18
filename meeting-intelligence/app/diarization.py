"""Speaker diarization via pyannote.audio (pyannote/speaker-diarization-3.1).

Requires HF_TOKEN in the repo-root .env (a HuggingFace access token with the
gated pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1 model
licenses accepted). The pipeline is downloaded once on first use and cached
locally by huggingface_hub — every run after that is fully offline.

Defaults to CPU: the two live-call bots already keep this machine's 8GB
RTX 3070 Ti near its VRAM ceiling, and meeting processing is a batch/offline
operation, not latency-critical, so CPU throughput is an acceptable tradeoff
for eliminating OOM risk on a live bot.
"""

import logging
import os

import torch
from pyannote.audio import Pipeline

log = logging.getLogger("meeting-intelligence")

_pipeline: Pipeline | None = None
_pipeline_device: str | None = None


class DiarizationUnavailable(RuntimeError):
    """Raised when HF_TOKEN is missing or the gated model can't be loaded."""


def get_pipeline(device: str = "cpu") -> Pipeline:
    global _pipeline, _pipeline_device
    if _pipeline is not None and _pipeline_device == device:
        return _pipeline

    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise DiarizationUnavailable(
            "HF_TOKEN is not set in .env — needed once to download the gated "
            "pyannote/speaker-diarization-3.1 model. See Live_meeting_capture.md §14a."
        )
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    except Exception as e:
        raise DiarizationUnavailable(f"Could not load pyannote pipeline: {e}") from e

    pipeline.to(torch.device(device))
    _pipeline, _pipeline_device = pipeline, device
    log.info("pyannote diarization pipeline loaded on %s", device)
    return _pipeline


def diarize(wav_path, device: str = "cpu") -> list[dict]:
    """Returns [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
    sorted by start time. Pure wrapper — timestamp alignment to Whisper
    segments happens in main_meet.py's processing pipeline, not here."""
    pipeline = get_pipeline(device)
    diarization = pipeline(str(wav_path))
    turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])
    return turns
