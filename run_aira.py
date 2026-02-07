import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from core.listener import record_audio
from core.stt import speech_to_text
from core.brain import think
from core.tts import speak

print("🤖 AIRA is running (always listening mode)")
print("👉 Just speak your question")

while True:
    record_audio(duration=4)   # shorter, less noise
    text = speech_to_text("input.wav")

    text = text.strip()
    if not text:
        continue

    print("📝 Heard:", text)

    response = think(text)
    print("🤖 AIRA:", response)

    speak(response)
