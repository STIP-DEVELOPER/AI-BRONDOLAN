import time

import serial

from src.services.arduino_commands import to_arduino_signal
from src.services.serial_autodetect import find_arduino_port


class SerialService:
    def __init__(self, port: str | None = None, baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def connect(self) -> bool:
        if self.ser and self.ser.is_open:
            return True

        if not self.port:
            self.port = find_arduino_port()

        if not self.port:
            print("[SERIAL] Port tidak ditemukan (autodetect gagal)")
            return False

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # tunggu Arduino ready
            print(f"[SERIAL] Connected to {self.port}")
            return True
        except serial.SerialException as exc:
            print(f"[SERIAL] Gagal konek ke {self.port}: {exc}")
            self.ser = None
            return False

    def send(self, message: str) -> bool:
        signal = to_arduino_signal(message)
        if not signal:
            print(f"[SERIAL] Perintah kosong, tidak dikirim: {message!r}")
            return False

        if self.ser and self.ser.is_open:
            self.ser.write(f"{signal}\n".encode())
            if signal != message.strip().upper():
                print(f"[SERIAL] Sent: {signal} ({message})")
            else:
                print(f"[SERIAL] Sent: {signal}")
            return True
        print(f"[SERIAL] Tidak terkirim (belum terhubung): {signal} ({message})")
        return False

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SERIAL] Closed")
        self.ser = None
