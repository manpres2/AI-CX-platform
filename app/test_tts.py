from kokoro import KPipeline
import soundfile as sf
import torch

print(f"CUDA available: {torch.cuda.is_available()}")

pipeline = KPipeline(lang_code='a')

text = "Hello, I am Manpreet's voice assistant. How can I help you today?"

print("Generating audio...")
for i, (gs, ps, audio) in enumerate(pipeline(text, voice='af_heart')):
    sf.write(f'C:\\AIManpres2\\logs\\test_output_{i}.wav', audio, 24000)
    print(f"Saved test_output_{i}.wav")

print("Done — check C:\\AIManpres2\\logs\\")