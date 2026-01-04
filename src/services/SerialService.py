import serial
import time


class SerialService:
    def __init__(self, port: str, baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def connect(self):
        if not self.port:
            return

        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        time.sleep(2)  # tunggu Arduino ready
        print(f"[SERIAL] Connected to {self.port}")

    def send(self, message: str):
        if self.ser and self.ser.is_open:
            self.ser.write(f"{message}\n".encode())
            print(f"[SERIAL] Sent: {message}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SERIAL] Closed")
