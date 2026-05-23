import os
import sys

import cv2

from src.config.performance import CAMERA_HEIGHT, CAMERA_WIDTH, MAX_CAMERA_PROBE


def _capture_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def probe_camera(index: int) -> bool:
    """Cek apakah index kamera bisa dibuka dan membaca frame."""
    cap = cv2.VideoCapture(index, _capture_backend())
    try:
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        ret, frame = cap.read()
        return ret and frame is not None and frame.size > 0
    finally:
        cap.release()


def list_available_cameras(max_probe: int | None = None) -> list[int]:
    if max_probe is None:
        max_probe = MAX_CAMERA_PROBE
    """Scan index 0..max_probe-1, kembalikan daftar kamera yang aktif."""
    found: list[int] = []
    for index in range(max_probe):
        if probe_camera(index):
            found.append(index)
    return found


def _env_camera_index(role: str) -> int | None:
    key = "CAMERA_NGINTIL" if role == "ngintil" else "CAMERA_BRONDOL"
    raw = os.getenv(key, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print(f"[CAMERA] {key} tidak valid: {raw}")
        return None


def assign_camera_indices(
    available: list[int] | None = None,
) -> tuple[int | None, int | None]:
    """
    Tetapkan index kamera untuk ngintil dan brondol.
    - Kamera 1 terdeteksi -> ngintil
    - Kamera 2 terdeteksi -> brondol (berbeda dari ngintil)
    - Override opsional via .env
    """
    if available is None:
        available = list_available_cameras()

    ngintil_override = _env_camera_index("ngintil")
    brondol_override = _env_camera_index("brondol")

    ngintil_index: int | None = None
    brondol_index: int | None = None

    if ngintil_override is not None:
        ngintil_index = ngintil_override
    elif available:
        ngintil_index = available[0]

    if brondol_override is not None:
        brondol_index = brondol_override
    elif len(available) >= 2:
        brondol_index = available[1]
    # Jika hanya 1 kamera: brondol tetap None (tidak berbagi)

    if (
        ngintil_index is not None
        and brondol_index is not None
        and ngintil_index == brondol_index
        and len(available) >= 2
    ):
        brondol_index = available[1]

    return ngintil_index, brondol_index
