"""Pemetaan perintah robot ke sinyal serial Arduino (1 karakter)."""

# Nama perintah aplikasi -> karakter Arduino
ARDUINO_COMMAND_MAP: dict[str, str] = {
    "RIGHT": "R",
    "LEFT": "L",
    "BACK": "B",
    "BACKWARD": "B",
    "MUNDUR": "B",
    "KIRI": "L",
    "KANAN": "R",
    "FORWARD": "F",
    "MAJU": "F",
    "STOP": "S",
    "BERHENTI": "S",
    "TAKE": "T",
    "AMBIL": "T",
    "CHECK": "C",
    "CEK": "C",
    # Sudah berupa sinyal Arduino
    "R": "R",
    "L": "L",
    "B": "B",
    "F": "F",
    "S": "S",
    "T": "T",
    "C": "C",
}


def to_arduino_signal(command: str) -> str:
    """
    Ubah perintah (RIGHT, FORWARD, TAKE, ...) menjadi 1 karakter untuk Arduino.
    Jika tidak dikenali, kembalikan uppercase as-is.
    """
    key = command.strip().upper()
    if not key:
        return ""

    if key in ARDUINO_COMMAND_MAP:
        return ARDUINO_COMMAND_MAP[key]

    return key
