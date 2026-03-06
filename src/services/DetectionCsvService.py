import csv
import os
from datetime import datetime, timezone
from typing import Optional


class DetectionCsvService:
    def __init__(self, file_path: str = "logs/detections.csv"):
        self._file_path = file_path
        self._current_id: int = 0
        self._ensure_directory()
        self._initialize_file_and_id()

    def _ensure_directory(self) -> None:
        directory = os.path.dirname(self._file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _initialize_file_and_id(self) -> None:
        if not os.path.exists(self._file_path):
            # Buat file baru dengan header
            with open(self._file_path, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["id", "created_at", "total_objects"])
            self._current_id = 0
            return

        # Jika file sudah ada, baca id terakhir (jika ada data)
        last_id = self._read_last_id()
        self._current_id = last_id if last_id is not None else 0

    def _read_last_id(self) -> Optional[int]:
        try:
            with open(self._file_path, mode="r", newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                # Lewati header
                next(reader, None)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row and last_row[0].isdigit():
                    return int(last_row[0])
        except FileNotFoundError:
            return None
        except Exception as exc:
            print(f"[CSV] Gagal membaca id terakhir: {exc}")
        return None

    def log_detection(self, total_objects: int) -> None:
        if total_objects is None:
            return

        created_at = datetime.now(timezone.utc).isoformat()
        self._current_id += 1

        try:
            with open(self._file_path, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self._current_id, created_at, total_objects])
        except Exception as exc:
            print(f"[CSV] Gagal menulis detection: {exc}")

