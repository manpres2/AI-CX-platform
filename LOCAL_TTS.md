# Local expressive TTS

In any voice bot admin panel, open Voice & Model > Text to Speech > Local,
select **Chatterbox (Expressive)**, and click **Test**.
Start with expressiveness 0.7 and pacing guidance 0.3. Click **Save TTS Settings**
to use the selection for calls and retain it after restarting.

Original Chatterbox provides English speech with an exaggeration control.
It is not a Kokoro model and does not expose named happy/sad emotion presets.
Its bundled voice works without a reference recording.
Official model: https://huggingface.co/ResembleAI/chatterbox

## Setup

Run `powershell -ExecutionPolicy Bypass -File setup_local_tts.ps1` at the
repository root with `uv` installed. This installs Python 3.11, CUDA PyTorch 2.6,
Kokoro, Chatterbox, and the English language dependency, then downloads the five
Chatterbox model files (about 3.2 GB). Python 3.12 is incompatible with the
NumPy constraint in Chatterbox 0.1.6. Setuptools is capped because Perth uses
pkg_resources. Weights and virtual environments stay out of Git.

The first speech request automatically starts the hidden loopback service
on port 8021. Logs: `local_tts_server.log`. Restart existing bot processes after
updating, so they route Kokoro through this service too.

## GPU ownership

One shared service owns both engines across all bot processes. Generation and
selection use one lock; loading an engine releases the previous model and CUDA
cache first. Changing Kokoro language or repository also replaces its model.
Changing the dropdown releases an inactive model without loading the new one.
Saving persists the choice; testing loads it on demand.

A concurrent generation/switch returns a busy error. Retry after audio finishes.
Only CUDA is supported; there is no CPU speech-model fallback. Whisper, Ollama,
and other applications have their own GPU allocations.

Run only one local service worker. Diagnostic endpoints:
`GET http://127.0.0.1:8021/health`, `POST /unload`.
The service is for local use and must remain bound to loopback.

## Validation

```powershell
.venv-local-tts/Scripts/python.exe -m unittest discover -s tests -p test_local_tts.py -v
venv/Scripts/python.exe -m unittest discover -s tests -p test_tts_providers.py -v
node tests/test_tts_ui.cjs
```

Live RTX 3070 Ti checks generated PCM for both engines and verified unload between
them. Chatterbox warm generation was about 4 seconds for a short preview.
First launch/model load is slower, and switching back must load the model again.
