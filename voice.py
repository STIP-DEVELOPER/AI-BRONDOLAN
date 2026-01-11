import os
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =====================
# CONFIG
# =====================
SAMPLE_RATE = 16000
CHANNELS = 1

VOICE_THRESHOLD = 0.003     # cocok mic laptop lama
SILENCE_DURATION = 1.0     # detik diam = stop rekam
MAX_RECORD_TIME = 10       # safety limit

AUDIO_INPUT = "input.wav"
AUDIO_OUTPUT = "reply.wav"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

audio_queue = queue.Queue()

# =====================
# AUDIO CALLBACK
# =====================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# =====================
# RECORD WITH VAD
# =====================
def record_until_silence():
    print("🎙️ Listening...")

    frames = []
    silence_start = None
    start_time = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        dtype="float32"
    ):
        while True:
            data = audio_queue.get()
            volume = np.linalg.norm(data)

            if volume > VOICE_THRESHOLD:
                silence_start = None
                frames.append(data)
            else:
                if frames:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        break

            if time.time() - start_time > MAX_RECORD_TIME:
                break

    if not frames:
        print("⚠️ Tidak ada suara")
        return False

    audio = np.concatenate(frames, axis=0)
    duration = len(audio) / SAMPLE_RATE

    if duration < 0.2:
        print(f"⚠️ Audio terlalu pendek ({duration:.2f}s)")
        return False

    sf.write(AUDIO_INPUT, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"✅ Voice captured ({duration:.2f}s)")
    return True

# =====================
# SPEECH → TEXT
# =====================
def speech_to_text():
    with open(AUDIO_INPUT, "rb") as f:
        result = client.audio.transcriptions.create(
            file=f,
            model="whisper-1"
        )
    return result.text.strip()

# =====================
# COMMAND PARSER (FUNCTION CALLING LIGHT)
# =====================
def parse_command(text: str):
    text = text.lower()

    if "maju" in text or "forward" in text:
        return "F"
    if "mundur" in text or "backward" in text:
        return "B"
    if "stop" in text or "berhenti" in text:
        return "S"

    return None

# =====================
# CHAT
# =====================
def chat(text, messages):
    messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

# =====================
# TEXT → SPEECH (HUMAN-LIKE)
# =====================
def text_to_speech(text):
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",   # paling natural & humble
        input=text
    )

    with open(AUDIO_OUTPUT, "wb") as f:
        f.write(response.read())

    data, samplerate = sf.read(AUDIO_OUTPUT, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()

# =====================
# MAIN LOOP (REALTIME)
# =====================
def main():
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah voice assistant robot yang ramah, singkat, "
                "dan fokus pada perintah MAJU, MUNDUR, STOP."
            )
        }
    ]

    print("🔊 Real-time Voice Agent")
    print("Ucapkan: MAJU / MUNDUR / STOP")
    print("Ctrl+C untuk keluar\n")

    try:
        while True:
            ok = record_until_silence()
            if not ok:
                continue

            text = speech_to_text()
            if not text:
                continue

            print(f"🗣️ Kamu: {text}")

            command = parse_command(text)
            if command:
                print(f"🛠️ COMMAND → {command}")
                # NANTI: kirim ke Arduino via serial
                # serial.write(command.encode())

                reply = f"Perintah {text.upper()} diterima"
            else:
                reply = chat(text, messages)

            print(f"🤖 Bot: {reply}")
            text_to_speech(reply)

    except KeyboardInterrupt:
        print("\n👋 Keluar")

if __name__ == "__main__":
    main()
