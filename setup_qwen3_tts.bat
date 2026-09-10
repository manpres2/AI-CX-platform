@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-qwen3tts\Scripts\python.exe" (
  py -3.12 -m venv .venv-qwen3tts
  if errorlevel 1 exit /b 1
)

".venv-qwen3tts\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv-qwen3tts\Scripts\python.exe" -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 exit /b 1
".venv-qwen3tts\Scripts\python.exe" -m pip install -r requirements-qwen3-tts.txt
if errorlevel 1 exit /b 1

echo.
".venv-qwen3tts\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 exit /b 1
echo Qwen3-TTS is installed. Run start_qwen3_tts.bat to start the service.
