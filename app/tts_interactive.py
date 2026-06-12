from kokoro import KPipeline
import soundfile as sf
import torch
import os

print("Loading model... (one time only)")
pipeline = KPipeline(lang_code='a')
voice = 'af_heart'
print(f"Model ready. Using voice: {voice}")
print("Type text and press Enter to hear it. Type 'quit' to exit.\n")

counter = 0
while True:
    text = input("Text> ").strip()
    if text.lower() == 'quit':
        break
    if not text:
        continue
    
    for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice)):
        out_path = f'C:\\AIManpres2\\logs\\out_{counter}.wav'
        sf.write(out_path, audio, 24000)
        counter += 1

    os.startfile(out_path)
    print(f"Playing: {out_path}\n")