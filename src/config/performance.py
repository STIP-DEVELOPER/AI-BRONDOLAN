"""Pengaturan performa — bisa di-override lewat .env"""

import os


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return max(0.1, float(raw))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# Model dilatih di 640; turunkan ke 320 di .env jika perlu hemat CPU
INFER_IMGSZ = _env_int("INFER_IMGSZ", 640)
INFER_CONF = _env_float("INFER_CONF", 0.5)
INFER_IOU = _env_float("INFER_IOU", 0.45)
MAX_DISPLAY_BOXES = _env_int("MAX_DISPLAY_BOXES", 5)
# Jalankan deteksi setiap N frame (sisanya pakai hasil terakhir)
INFER_EVERY_N_FRAMES = _env_int("INFER_EVERY_N_FRAMES", 3)

# Resolusi capture kamera
CAMERA_WIDTH = _env_int("CAMERA_WIDTH", 424)
CAMERA_HEIGHT = _env_int("CAMERA_HEIGHT", 240)
CAMERA_SCAN_INTERVAL_SEC = _env_float("CAMERA_SCAN_INTERVAL_SEC", 5.0)
CAMERA_SCAN_INTERVAL_CONNECTED_SEC = _env_float("CAMERA_SCAN_INTERVAL_CONNECTED_SEC", 12.0)
MAX_CAMERA_PROBE = _env_int("MAX_CAMERA_PROBE", 6)

# Batasi kecepatan loop UI (~20 FPS tampilan)
DISPLAY_WAIT_MS = _env_int("DISPLAY_WAIT_MS", 50)

# Dashboard: deteksi ngintil & brondol bergantian (tidak bersamaan tiap frame)
UI_ALTERNATE_INFER = _env_bool("UI_ALTERNATE_INFER", True)

# Resize frame sebelum infer (lebar maks)
INFER_MAX_FRAME_WIDTH = _env_int("INFER_MAX_FRAME_WIDTH", 424)

# Jendela OpenCV fullscreen saat aplikasi dijalankan
APP_FULLSCREEN = _env_bool("APP_FULLSCREEN", True)
