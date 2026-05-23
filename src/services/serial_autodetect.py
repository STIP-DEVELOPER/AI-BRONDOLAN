import os
import sys

from serial.tools import list_ports

# VID umum Arduino & adapter USB-serial
_ARDUINO_VIDS = {
    0x2341,  # Arduino SA
    0x2A03,  # Arduino LLC
    0x1B4F,  # SparkFun
    0x239A,  # Adafruit
    0x16C0,  # Teensy
    0x27DB,  # Arduino SA (native USB)
}

_USB_SERIAL_VIDS = {
    0x10C4,  # Silicon Labs CP210x
    0x0403,  # FTDI
    0x1A86,  # WCH CH340
    0x7523,  # Prolific / CH340 clone
}

_KEYWORDS = (
    "arduino",
    "usbmodem",
    "usbserial",
    "usb serial",
    "ch340",
    "cp210",
    "ft232",
    "ftdi",
    "wch",
    "serial",
    "uart",
    "mega",
    "uno",
    "leonardo",
)


def _port_score(port) -> int:
    desc = (port.description or "").lower()
    manufacturer = (port.manufacturer or "").lower()
    hwid = (port.hwid or "").lower()
    device = (port.device or "").lower()
    combined = " ".join((desc, manufacturer, hwid, device))

    if "bluetooth" in combined:
        return -100

    score = 0
    if port.vid in _ARDUINO_VIDS:
        score += 100
    elif port.vid in _USB_SERIAL_VIDS:
        score += 50

    if any(keyword in combined for keyword in _KEYWORDS):
        score += 30

    # macOS: /dev/cu.* lebih stabil untuk kirim data
    if sys.platform == "darwin" and device.startswith("/dev/cu."):
        score += 10

    if device.startswith("/dev/ttyusb") or device.startswith("/dev/ttyacm"):
        score += 10

    return score


def find_arduino_port() -> str | None:
    """
    Cari port serial Arduino/USB-serial yang terhubung.
    Urutan: override .env (opsional) -> skor tertinggi dari port terdeteksi.
    """
    override = os.getenv("SERIAL_PORT", "").strip()
    if override:
        return override

    candidates: list[tuple[int, str, str]] = []

    for port in list_ports.comports():
        score = _port_score(port)
        if score <= 0:
            continue
        label = port.description or port.device
        candidates.append((score, port.device, label))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    print(f"[SERIAL] Autodetect: {best[1]} ({best[2]})")
    return best[1]
