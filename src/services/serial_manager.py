import os
import threading

from src.services.SerialService import SerialService
from src.services.serial_autodetect import find_arduino_port

_lock = threading.Lock()
_shared: SerialService | None = None
_refcount = 0


def _default_baudrate() -> int:
    return int(os.getenv("SERIAL_BAUDRATE", os.getenv("BAUDRATE", "9600")))


def get_shared_serial(baudrate: int | None = None) -> SerialService | None:
    """
    Ambil koneksi serial bersama (satu port untuk seluruh modul dalam proses ini).
    Panggil release_shared_serial() saat selesai.
    """
    global _shared, _refcount

    rate = baudrate if baudrate is not None else _default_baudrate()

    with _lock:
        if _shared is not None and _shared.ser and _shared.ser.is_open:
            _refcount += 1
            return _shared

        port = find_arduino_port()
        if not port:
            print("[SERIAL] Arduino tidak ditemukan. Colokkan USB lalu jalankan ulang.")
            return None

        service = SerialService(port, rate)
        if not service.connect():
            return None

        _shared = service
        _refcount = 1
        return _shared


def release_shared_serial() -> None:
    """Turunkan refcount; tutup port jika tidak ada yang memakai lagi."""
    global _shared, _refcount

    with _lock:
        if _refcount <= 0:
            return
        _refcount -= 1
        if _refcount > 0:
            return
        if _shared is not None:
            _shared.close()
            _shared = None


def is_serial_connected() -> bool:
    with _lock:
        return _shared is not None and _shared.ser is not None and _shared.ser.is_open
