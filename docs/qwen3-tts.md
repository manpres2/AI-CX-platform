# Local and online TTS

In Voice & LLM, select **Local** and then **Kokoro** or **Qwen3-TTS**.
Select **API (Online)** for ElevenLabs, OpenAI, or a hosted Veena endpoint.
Use Test to preview the selected provider, then save the voice/model settings.

Qwen3 supports built-in voices (Ryan, Aiden, Vivian, Serena, Uncle Fu, Dylan,
Eric, Ono Anna, and Sohee) or My Voice using a reference recording and transcript.
Ryan and Aiden are native English voices.

Run setup_qwen3_tts.bat once, then start_qwen3_tts.bat. The service listens on
127.0.0.1:8020. Model weights download on the first synthesis request, not at
service startup. A health response confirms the service is reachable; it does
not confirm weights are downloaded or synthesis is fast enough for live calls.

The isolated environment avoids changing the existing Kokoro dependencies.
Check torch.cuda.is_available() in that environment before expecting GPU speed.
The service requires CUDA and does not fall back to CPU. The setup script installs
the official PyTorch 2.6.0 / TorchAudio 2.6.0 CUDA 12.4 builds and verifies the GPU.
Local Qwen3 still uses an HTTP connection internally.

The default built-in model is Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice; cloning uses
Qwen/Qwen3-TTS-12Hz-0.6B-Base. Only one model stays loaded at a time.
QWEN3_TTS_PRESET_MODEL and QWEN3_TTS_MODEL override these respectively.
QWEN3_TTS_DEVICE overrides the device and QWEN3_TTS_API_KEY enables bearer auth.

Only the selected TTS engine is retained by a bot: Kokoro loads lazily and is
released before Qwen3 synthesis; choosing Kokoro releases the shared Qwen3 model.
Saving an online TTS selection releases both. Previews also switch residency.
Model loading/unloading and synthesis are serialized within each bot. Whisper,
Ollama, other running bots, and other GPU applications have their own allocations.
The health endpoints report kokoro_loaded and Qwen3 GPU allocation respectively.

Existing saved Qwen3 configurations migrate from API mode to Local automatically.
After updating backend code, restart running bots and refresh the admin page.

Tests: run the provider contract tests with the project virtual environment:

    venv\Scripts\python.exe -m unittest discover -s tests -p test_tts_providers.py -v

These tests use a mock speech model to verify HTTP, selection, and PCM contracts.
They do not evaluate generated speech quality or real-model latency.
