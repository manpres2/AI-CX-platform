"""
Veena TTS — Modal deployment
=============================
Serves maya-research/veena-tts (Hindi/English/code-mixed, 24kHz, 4 voices:
kavya/agastya/maitri/vinaya) as an HTTP endpoint that the bot's admin panel
can point at (Voice & Model -> Cloud TTS Engine -> Veena (Modal)).

Setup (one-time):
    pip install modal
    modal setup
    modal secret create veena-tts-key VEENA_API_KEY=<pick-a-random-string>   # optional but recommended

Deploy:
    modal deploy modal_veena_tts.py

Deploy prints a URL like:
    https://<workspace>--veena-tts-tts.modal.run
Paste that (plus the API key you chose above, if any) into the admin panel.

Notes:
  - First request after idle has a cold-start delay while the model loads
    onto a fresh GPU container (scaledown_window below controls how long a
    warm container is kept around after the last request).
  - Default GPU is A10G. For lower latency, change gpu="A10G" to "A100" or
    "H100" below (higher $/hr).
  - This is the same Modal app as the one in the repo root / bot-template —
    you only need to deploy it ONCE. Every bot's admin panel can point at
    the same endpoint URL; this copy exists just so the "modal deploy
    modal_veena_tts.py" instruction works from this folder too.
"""

import os

import modal
from fastapi import HTTPException
from fastapi.responses import Response

app = modal.App("veena-tts")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2",
    "torchaudio>=2.2",
    "transformers>=4.44",
    "accelerate",
    "bitsandbytes",
    "snac",
    "soundfile",
    "numpy",
    "fastapi[standard]",
)

try:
    auth_secret = [modal.Secret.from_name("veena-tts-key")]
except Exception:
    auth_secret = []  # no secret created -> endpoint runs unauthenticated

START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266
SPEAKERS = {"kavya", "agastya", "maitri", "vinaya"}


@app.cls(gpu="A10G", image=image, scaledown_window=300, secrets=auth_secret)
class Veena:
    @modal.enter()
    def load(self):
        import torch
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "maya-research/veena-tts",
            quantization_config=quant,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "maya-research/veena-tts", trust_remote_code=True
        )
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
        self.torch = torch

    @modal.method()
    def synthesize(
        self, text: str, speaker: str = "kavya", temperature: float = 0.4, top_p: float = 0.9
    ) -> bytes:
        torch = self.torch
        speaker = speaker if speaker in SPEAKERS else "kavya"

        prompt_tokens = self.tokenizer.encode(f"<spk_{speaker}> {text}", add_special_tokens=False)
        input_tokens = [
            START_OF_HUMAN_TOKEN,
            *prompt_tokens,
            END_OF_HUMAN_TOKEN,
            START_OF_AI_TOKEN,
            START_OF_SPEECH_TOKEN,
        ]
        input_ids = torch.tensor([input_tokens], device=self.model.device)
        max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
            )

        generated_ids = output[0][len(input_tokens):].tolist()
        snac_tokens = [
            t
            for t in generated_ids
            if AUDIO_CODE_BASE_OFFSET <= t < AUDIO_CODE_BASE_OFFSET + 7 * 4096
        ]
        snac_tokens = snac_tokens[: len(snac_tokens) - (len(snac_tokens) % 7)]
        if not snac_tokens:
            return b""

        codes_lvl = [[], [], []]
        offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]
        for i in range(0, len(snac_tokens), 7):
            codes_lvl[0].append(snac_tokens[i] - offsets[0])
            codes_lvl[1].append(snac_tokens[i + 1] - offsets[1])
            codes_lvl[1].append(snac_tokens[i + 4] - offsets[4])
            codes_lvl[2].append(snac_tokens[i + 2] - offsets[2])
            codes_lvl[2].append(snac_tokens[i + 3] - offsets[3])
            codes_lvl[2].append(snac_tokens[i + 5] - offsets[5])
            codes_lvl[2].append(snac_tokens[i + 6] - offsets[6])

        device = next(self.snac.parameters()).device
        hierarchical_codes = []
        for lvl in codes_lvl:
            t = torch.tensor(lvl, dtype=torch.int32, device=device).unsqueeze(0)
            t = torch.clamp(t, 0, 4095)
            hierarchical_codes.append(t)

        with torch.no_grad():
            audio_hat = self.snac.decode(hierarchical_codes)

        import numpy as np

        audio = audio_hat.squeeze().clamp(-1, 1).cpu().numpy()
        return (audio * 32767).astype(np.int16).tobytes()


@app.function(image=image, secrets=auth_secret)
@modal.fastapi_endpoint(method="POST")
def tts(item: dict):
    """POST {"text": "...", "speaker": "kavya", "api_key": "..."}
    -> raw 16-bit PCM mono @ 24kHz (matches the bot's SAMPLE_RATE, no
    resampling needed on the receiving end)."""
    required_key = os.environ.get("VEENA_API_KEY", "")
    if required_key and item.get("api_key") != required_key:
        raise HTTPException(status_code=401, detail="invalid api_key")

    text = (item.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    pcm = Veena().synthesize.remote(text, item.get("speaker", "kavya"))
    return Response(content=pcm, media_type="audio/L16")
