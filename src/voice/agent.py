import os
import queue
import re
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

from src.services.arduino_commands import to_arduino_signal
from src.services.SerialService import SerialService
from src.services.serial_manager import get_shared_serial, release_shared_serial

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
VOICE_THRESHOLD = 0.003
SILENCE_DURATION = 1.0
MAX_RECORD_TIME = 10
AUDIO_INPUT = "input.wav"
AUDIO_OUTPUT = "reply.wav"

# TTS: gpt-4o-mini-tts — jernih + natural; pemutaran native tanpa slow-down/gain berat.
TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "coral")
TTS_SPEED = float(os.getenv("VOICE_TTS_SPEED", "1.0"))
PAUSE_BEFORE_SPEAK = float(os.getenv("VOICE_PAUSE_BEFORE_SPEAK", "0.05"))
PLAYBACK_GAIN = float(os.getenv("VOICE_PLAYBACK_GAIN", "1.0"))
REPLY_MAX_WORDS = int(os.getenv("VOICE_REPLY_MAX_WORDS", "20"))
MAX_CHAT_HISTORY = int(os.getenv("VOICE_CHAT_HISTORY", "8"))
FALLBACK_INDONESIAN = "Maaf, saya hanya bisa menjawab dalam Bahasa Indonesia."

# Deteksi teks bukan Bahasa Indonesia (aksara asing / dominan Inggris)
_FOREIGN_SCRIPT_RE = re.compile(
    r"[\u0370-\u03FF\u0400-\u04FF\u0600-\u06FF\u0900-\u097F"
    r"\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]"
)
_ENGLISH_HEAVY_RE = re.compile(
    r"\b(the|is|are|was|were|have|has|had|will|would|could|should|this|that|"
    r"these|those|you|your|what|how|why|when|where|who|which|with|from|about|"
    r"hello|please|thank|thanks|yes|no|okay|ok|my|name|i|am|it's|don't)\b",
    re.I,
)
_INDONESIAN_MARKERS_RE = re.compile(
    r"\b(saya|kamu|anda|yang|dan|atau|adalah|tidak|ini|itu|dengan|untuk|pada|"
    r"dari|ke|di|akan|bisa|ada|apa|siapa|nama|baik|maaf|robot|brondolan|"
    r"kelapa|sawit|perintah|jawab|halo|terima|kasih)\b",
    re.I,
)
_NAME_QUESTION_RE = re.compile(
    r"(nama\s*(kamu|anda|mu|nya)|siapa\s*(kamu|anda|namamu|nama\s*mu)|"
    r"panggilan\s*mu|identitas|what\s*('s|is)\s*your\s*name|who\s*are\s*you)",
    re.I,
)

TTS_INSTRUCTIONS = os.getenv(
    "VOICE_TTS_INSTRUCTIONS",
    (
        "Bicara seperti manusia Indonesia yang ramah: nada hangat, natural, "
        "bukan suara robot atau metalik. Artikulasi jelas, tempo percakapan normal, "
        "mudah didengar. Hindari terlalu cepat, terlalu pelan, atau monoton."
    ),
)

COMMAND_LABELS = {
    "FORWARD": "MAJU",
    "BACK": "MUNDUR",
    "LEFT": "KIRI",
    "RIGHT": "KANAN",
    "STOP": "STOP",
    "TAKE": "AMBIL",
    "CHECK": "CHECK",
}

SYSTEM_PROMPT = (
    "Namamu adalah SAVIRA BRONDOLAN. Kamu asisten suara robot panen brondolan (buah kelapa sawit) yang ramah. "
    "Jika ditanya nama atau siapa kamu, jawab singkat bahwa kamu BRONDOLAN. "
    "ATURAN BAHASA (wajib):\n"
    "- Jawab HANYA dalam Bahasa Indonesia yang baik dan benar.\n"
    "- Jangan pernah memakai Inggris, Spanyol, Mandarin, Jepang, atau bahasa lain.\n"
    "- Meskipun pengguna bertanya dalam bahasa asing, tetap jawab dalam Bahasa Indonesia.\n"
    "SINGKAT (wajib): Satu kalimat saja, maksimal 20 kata. "
    "Tanpa daftar, poin, atau penjelasan panjang. "
    "Jawab inti pertanyaan saja dengan kata sesedikit mungkin, gaya lisan alami seperti bicara langsung.\n"
    "Hanya gunakan huruf Latin biasa (a-z), tanpa aksara Cina, Jepang, Korea, Arab, atau Cyrillic.\n"
    "Boleh menjelaskan topik umum secara singkat jika ditanya. "
    "Jika pengguna meminta robot bergerak (maju, mundur, kiri, kanan, stop, ambil, check), "
    "jangan mengulangi bahwa perintah sedang dikirim—itu ditangani sistem kontrol terpisah."
)


class VoiceAgent:
    def __init__(
        self,
        serial_service: SerialService | None = None,
        manage_serial: bool = True,
    ):
        self._lock = threading.Lock()
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._manage_serial = manage_serial

        self.status = "Siap"
        self.user_text = ""
        self.bot_text = ""
        self.command: str | None = None
        self.volume = 0.0
        self.muted = False
        self.serial_connected = False

        self._serial = serial_service
        self._connect_serial()

        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _connect_serial(self) -> None:
        if self._serial is None and self._manage_serial:
            self._serial = get_shared_serial()

        if self._serial is None:
            self.serial_connected = False
            return

        self.serial_connected = self._serial.connect()

    def _set_state(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "user_text": self.user_text,
                "bot_text": self.bot_text,
                "command": self.command,
                "volume": self.volume,
                "muted": self.muted,
                "serial_connected": self.serial_connected,
            }

    def is_muted(self) -> bool:
        with self._lock:
            return self.muted

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self.muted = muted
            if muted:
                self.status = "Dibisukan — klik UNMUTE"
                self.volume = 0.0
            else:
                self.status = "Siap — ucapkan perintah"

        if muted:
            self._drain_audio_queue()

    def toggle_mute(self) -> bool:
        self.set_muted(not self.is_muted())
        return self.is_muted()

    def _drain_audio_queue(self) -> None:
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _send_robot_command(self, command: str) -> bool:
        if not self._serial:
            print(f"[SERIAL] Perintah {command} tidak terkirim (Arduino tidak terdeteksi)")
            return False

        sent = self._serial.send(command)
        self.serial_connected = bool(
            self._serial.ser and self._serial.ser.is_open
        )
        return sent

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        volume = float(np.linalg.norm(indata))
        self._set_state(volume=volume)
        self._audio_queue.put(indata.copy())

    def _record_until_silence(self) -> bool:
        if self.is_muted():
            return False

        self._set_state(status="Mendengarkan...", volume=0.0)
        self._drain_audio_queue()

        frames = []
        silence_start = None
        start_time = time.time()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self._audio_callback,
            dtype="float32",
        ):
            while self._running:
                if self.is_muted():
                    return False

                try:
                    data = self._audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if self.is_muted():
                    return False

                volume = float(np.linalg.norm(data))
                self._set_state(volume=volume)

                if volume > VOICE_THRESHOLD:
                    silence_start = None
                    frames.append(data)
                elif frames:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        break

                if time.time() - start_time > MAX_RECORD_TIME:
                    break

        if not frames:
            self._set_state(status="Tidak ada suara")
            return False

        audio = np.concatenate(frames, axis=0)
        duration = len(audio) / SAMPLE_RATE
        if duration < 0.2:
            self._set_state(status="Audio terlalu pendek")
            return False

        sf.write(AUDIO_INPUT, audio, SAMPLE_RATE, subtype="PCM_16")
        self._set_state(status=f"Terekam ({duration:.1f}s)")
        return True

    def _speech_to_text(self) -> str:
        self._set_state(status="Transkripsi...")
        with open(AUDIO_INPUT, "rb") as f:
            result = self._client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
                language="id",
                prompt=(
                    "BRONDOLAN. Bahasa Indonesia. "
                    "Perintah robot: maju, mundur, kiri, kanan, stop, ambil, check."
                ),
            )
        return result.text.strip()

    @staticmethod
    def _parse_command(text: str) -> str | None:
        normalized = text.lower().strip()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        tokens = set(normalized.split())

        stop_words = {"stop", "berhenti", "henti", "diam"}
        kiri_words = {"kiri", "left"}
        kanan_words = {"kanan", "right"}
        mundur_words = {"mundur", "backward", "belakang", "back"}
        maju_words = {"maju", "forward", "depan"}
        take_words = {"ambil", "take", "pick", "grab", "petik"}
        check_words = {"check", "cek", "periksa", "scan"}

        # Urutan: perintah spesifik dulu (ambil/check) sebelum gerak umum
        if tokens & take_words or re.search(r"\b(ambil|take)\b", normalized):
            return "TAKE"
        if tokens & check_words or re.search(r"\b(check|cek)\b", normalized):
            return "CHECK"
        if tokens & stop_words or re.search(r"\bstop\b", normalized):
            return "STOP"
        if tokens & kiri_words or re.search(r"\b(kiri|left)\b", normalized):
            return "LEFT"
        if tokens & kanan_words or re.search(r"\b(kanan|right)\b", normalized):
            return "RIGHT"
        if tokens & mundur_words or re.search(r"\b(mundur|back)\b", normalized):
            return "BACK"
        if tokens & maju_words or re.search(r"\b(maju|forward)\b", normalized):
            return "FORWARD"

        return None

    @staticmethod
    def _is_name_question(text: str) -> bool:
        normalized = text.lower().strip()
        if not normalized:
            return False
        if "brondolan" in normalized and re.search(r"\b(nama|siapa|name)\b", normalized):
            return True
        return bool(_NAME_QUESTION_RE.search(normalized))

    @staticmethod
    def _sanitize_indonesian(text: str) -> str:
        cleaned = _FOREIGN_SCRIPT_RE.sub("", text or "")
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _looks_indonesian(text: str) -> bool:
        text = (text or "").strip()
        if not text or _FOREIGN_SCRIPT_RE.search(text):
            return False

        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False

        latin = sum(1 for c in letters if ord(c) < 128)
        if latin / len(letters) < 0.95:
            return False

        en_hits = len(_ENGLISH_HEAVY_RE.findall(text))
        id_hits = len(_INDONESIAN_MARKERS_RE.findall(text))
        words = text.split()

        if en_hits >= 2 and id_hits == 0:
            return False
        if len(words) >= 4 and en_hits > id_hits and en_hits >= 2:
            return False

        return True

    def _trim_messages(self) -> None:
        if not self._messages:
            return
        system = self._messages[0]
        rest = self._messages[1:]
        if len(rest) > MAX_CHAT_HISTORY:
            rest = rest[-MAX_CHAT_HISTORY:]
        self._messages = [system, *rest]

    def _chat_indonesian_retry(self, user_text: str) -> str:
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Pertanyaan pengguna: {user_text}\n\n"
                        "Jawab HANYA satu kalimat Bahasa Indonesia (maksimal 20 kata). "
                        "Dilarang memakai Inggris atau bahasa lain."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=60,
        )
        return response.choices[0].message.content or ""

    def _ensure_indonesian(self, reply: str, user_text: str) -> str:
        reply = self._clamp_reply(self._sanitize_indonesian(reply))
        if reply and self._looks_indonesian(reply):
            return reply

        retry = self._chat_indonesian_retry(user_text)
        retry = self._clamp_reply(self._sanitize_indonesian(retry))
        if retry and self._looks_indonesian(retry):
            return retry

        return FALLBACK_INDONESIAN

    def _handle_robot_command(self, command: str) -> str:
        label = COMMAND_LABELS.get(command, command)
        sent = self._send_robot_command(command)
        signal = to_arduino_signal(command)

        if sent:
            return f"Baik, {label} dikirim ke Arduino ({signal})."

        return f"{label} dikenali; koneksi serial Arduino belum aktif."

    @staticmethod
    def _clamp_reply(text: str, max_words: int = REPLY_MAX_WORDS) -> str:
        text = (text or "").strip()
        if not text:
            return text

        for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
            idx = text.find(sep)
            if idx != -1:
                text = text[: idx + 1].strip()
                break

        words = text.split()
        if len(words) <= max_words:
            return text

        shortened = " ".join(words[:max_words]).rstrip(",;:")
        if shortened and shortened[-1] not in ".!?":
            shortened += "."
        return shortened

    def _chat(self, text: str) -> str:
        if self._is_name_question(text):
            return "Nama saya BRONDOLAN."

        self._set_state(status="Memproses...")
        user_content = (
            f"{text}\n\n"
            "(Wajib: jawab HANYA Bahasa Indonesia, satu kalimat, maksimal 20 kata.)"
        )
        self._messages.append({"role": "user", "content": user_content})
        self._trim_messages()

        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self._messages,
            temperature=0.4,
            max_tokens=60,
        )
        reply = self._ensure_indonesian(
            response.choices[0].message.content or "", text
        )
        self._messages.append({"role": "assistant", "content": reply})
        return reply

    @staticmethod
    def _prepare_playback_audio(data: np.ndarray) -> np.ndarray:
        """Hanya naikkan volume jika terlalu pelan — jangan ubah audio yang sudah jelas."""
        if data.size == 0:
            return data

        audio = np.asarray(data, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-6:
            return audio

        if peak < 0.12:
            target = min(0.9 * PLAYBACK_GAIN, 0.9)
            audio = audio * (target / peak)

        return np.clip(audio, -1.0, 1.0)

    def _text_to_speech(self, text: str) -> None:
        if self.is_muted():
            return

        if PAUSE_BEFORE_SPEAK > 0:
            time.sleep(PAUSE_BEFORE_SPEAK)

        self._set_state(status="Memutar jawaban...")
        speech_kwargs: dict = {
            "model": TTS_MODEL,
            "voice": TTS_VOICE,
            "input": text,
            "response_format": "wav",
        }

        if TTS_MODEL.startswith("gpt-4o"):
            if TTS_INSTRUCTIONS.strip():
                speech_kwargs["instructions"] = TTS_INSTRUCTIONS.strip()
        else:
            speech_kwargs["speed"] = TTS_SPEED

        response = self._client.audio.speech.create(**speech_kwargs)
        with open(AUDIO_OUTPUT, "wb") as f:
            f.write(response.read())

        data, samplerate = sf.read(AUDIO_OUTPUT, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        data = self._prepare_playback_audio(data)

        sd.stop()
        sd.play(data, samplerate)
        sd.wait()

    def run_loop(self) -> None:
        self._running = True
        self._set_state(
            status="Siap — perintah: MAJU/MUNDUR/KIRI/KANAN/STOP/AMBIL/CHECK"
        )

        while self._running:
            if self.is_muted():
                time.sleep(0.2)
                continue

            if not self._record_until_silence():
                time.sleep(0.2)
                continue

            text = self._speech_to_text()
            if not text:
                self._set_state(status="Tidak terbaca")
                continue

            command = self._parse_command(text)
            if command:
                reply = self._ensure_indonesian(
                    self._handle_robot_command(command), text
                )
            elif self._is_name_question(text):
                command = None
                reply = "Nama saya BRONDOLAN."
            else:
                command = None
                reply = self._chat(text)

            self._set_state(
                user_text=text,
                bot_text=reply,
                command=command,
                status="Selesai",
            )
            self._text_to_speech(reply)
            self._set_state(
                status="Siap — perintah: MAJU/MUNDUR/KIRI/KANAN/STOP/AMBIL/CHECK",
                volume=0.0,
            )

    def stop(self) -> None:
        self._running = False
        if self._manage_serial:
            release_shared_serial()
        self.serial_connected = False
