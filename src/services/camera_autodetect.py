import os
import sys
import time
from pathlib import Path

import cv2

from src.config.performance import (
    CAMERA_HEIGHT,
    CAMERA_PROBE_DELAY_SEC,
    CAMERA_WIDTH,
    MAX_CAMERA_PROBE,
)

_PROBE_RETRIES = 2


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def capture_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if is_linux():
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def _probe_settle_delay() -> None:
    if sys.platform == "darwin" or is_linux():
        time.sleep(CAMERA_PROBE_DELAY_SEC)


def _device_path(index: int) -> str | None:
    if is_linux():
        path = f"/dev/video{index}"
        if os.path.exists(path):
            return path
    return None


def open_video_capture(index: int) -> cv2.VideoCapture:
    """Buka kamera by index; di Linux pakai path /dev/videoN + V4L2."""
    backend = capture_backend()
    path = _device_path(index)
    if path is not None:
        return cv2.VideoCapture(path, backend)
    return cv2.VideoCapture(index, backend)


def _read_test_frame(cap: cv2.VideoCapture) -> bool:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(3):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True
        time.sleep(0.05)
    return False


def _linux_video_indices(max_probe: int) -> list[int]:
    """Index dari /dev/video* (Linux Mint, V4L2)."""
    indices: list[int] = []
    for path in sorted(Path("/dev").glob("video*")):
        suffix = path.name[5:]
        if suffix.isdigit():
            idx = int(suffix)
            if idx < max_probe:
                indices.append(idx)
    return sorted(set(indices))


def _probe_order(max_probe: int) -> list[int]:
    """
    Urutan scan: di Linux utamakan index 0 dan 1 (/dev/video0, /dev/video1).
    """
    if is_linux():
        on_disk = _linux_video_indices(max_probe)
        order: list[int] = []
        for idx in (0, 1):
            if idx < max_probe and (idx in on_disk or not on_disk):
                order.append(idx)
        for idx in on_disk:
            if idx not in order:
                order.append(idx)
        for idx in range(max_probe):
            if idx not in order:
                order.append(idx)
        return order

    return list(range(max_probe))


def probe_camera(index: int) -> bool:
    """Cek apakah index kamera bisa dibuka dan membaca frame."""
    for attempt in range(_PROBE_RETRIES):
        cap = open_video_capture(index)
        try:
            if not cap.isOpened():
                continue
            if _read_test_frame(cap):
                return True
        finally:
            cap.release()
        _probe_settle_delay()
    return False


def _probe_pair_simultaneous(index_a: int, index_b: int) -> bool:
    """Kedua kamera harus bisa dibuka bersamaan (macOS & Linux)."""
    cap_a = open_video_capture(index_a)
    if not cap_a.isOpened():
        cap_a.release()
        return False

    cap_b = open_video_capture(index_b)
    if not cap_b.isOpened():
        cap_a.release()
        cap_b.release()
        return False

    ok = _read_test_frame(cap_a) and _read_test_frame(cap_b)
    cap_a.release()
    cap_b.release()
    _probe_settle_delay()
    return ok


def _ensure_dual_pair(found: list[int], max_probe: int) -> list[int]:
    if len(found) >= 2:
        return found

    if is_linux() and 0 not in found and _device_path(0):
        if probe_camera(0):
            found.append(0)
    if is_linux() and 1 not in found and _device_path(1):
        if probe_camera(1):
            found.append(1)

    found = sorted(set(found))
    if len(found) >= 2:
        return found

    for i in range(max_probe):
        for j in range(i + 1, max_probe):
            if i in found and j in found:
                continue
            if _probe_pair_simultaneous(i, j):
                return sorted(set(found) | {i, j})

    return sorted(set(found))


def list_available_cameras(
    max_probe: int | None = None,
    held_indices: list[int] | None = None,
) -> list[int]:
    """
    Scan index kamera yang aktif.

    held_indices: index yang sedang dipakai — jangan di-probe ulang
    (macOS/Linux: membuka device yang sama saat sudah terbuka sering gagal).
    """
    if max_probe is None:
        max_probe = MAX_CAMERA_PROBE

    held = set(held_indices or [])
    found: list[int] = []

    for index in _probe_order(max_probe):
        if index in held:
            found.append(index)
            continue
        if probe_camera(index):
            found.append(index)
        _probe_settle_delay()

    return _ensure_dual_pair(found, max_probe)


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
    Tetapkan index untuk ngintil (pertama) dan brondol (kedua).
    Di Linux standar: /dev/video0 -> ngintil, /dev/video1 -> brondol.
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

    if is_linux() and not brondol_override:
        if 0 in available and 1 in available:
            ngintil_index = 0
            brondol_index = 1
        elif ngintil_index == 0 and 1 in available:
            brondol_index = 1
        elif ngintil_index == 1 and 0 in available:
            ngintil_index = 0
            brondol_index = 1

    if (
        ngintil_index is not None
        and brondol_index is not None
        and ngintil_index == brondol_index
    ):
        for idx in available:
            if idx != ngintil_index:
                brondol_index = idx
                break

    return ngintil_index, brondol_index
