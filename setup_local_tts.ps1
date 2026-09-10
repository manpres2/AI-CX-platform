$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw "Python installation failed" }
if (-not (Test-Path .venv-local-tts/Scripts/python.exe)) {
    uv venv --python 3.11 .venv-local-tts
    if ($LASTEXITCODE -ne 0) { throw "Environment creation failed" }
}
uv pip install --python .venv-local-tts/Scripts/python.exe torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "CUDA installation failed" }
uv pip install --python .venv-local-tts/Scripts/python.exe chatterbox-tts==0.1.6 kokoro==0.9.4 fastapi uvicorn httpx 'setuptools<81' pip 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
if ($LASTEXITCODE -ne 0) { throw "TTS dependencies installation failed" }
& .venv-local-tts/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox', local_dir='models/chatterbox', allow_patterns=['ve.safetensors','t3_cfg.safetensors','s3gen.safetensors','tokenizer.json','conds.pt'])"
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }
