import os
import sys
import threading
import time
from itertools import combinations
from pathlib import Path

import cv2

from src.config.performance import (
    CAMERA_HEIGHT,
    CAMERA_OPEN_TIMEOUT_SEC,
    CAMERA_PROBE_DELAY_SEC,
    CAMERA_WIDTH,
    MAX_CAMERA_PROBE,
)

_PROBE_RETRIES = 2
_probe_lock = threading.Lock()

# V4L2 capability flags (linux/videodev2.h)
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_META_CAPTURE = 0x00800000


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def capture_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if is_linux():
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def _suppress_opencv_probe_noise() -> None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass


def _probe_settle_delay() -> None:
    if sys.platform == "darwin" or is_linux():
        time.sleep(CAMERA_PROBE_DELAY_SEC)


def _device_path(index: int) -> str | None:
    if is_linux():
        path = f"/dev/video{index}"
        if os.path.exists(path):
            return path
    return None


def _read_sysfs_caps(index: int) -> int | None:
    """Baca device_caps / capabilities dari sysfs (Linux 4.16+)."""
    base = Path(f"/sys/class/video4linux/video{index}")
    for name in ("device_caps", "capabilities"):
        cap_file = base / name
        if not cap_file.is_file():
            continue
        try:
            return int(cap_file.read_text().strip(), 16)
        except (OSError, ValueError):
            continue
    return None


def _linux_sysfs_has_device_caps() -> bool:
    """True jika kernel mengekspos device_caps (banyak laptop lama tidak)."""
    sys_dir = Path("/sys/class/video4linux")
    if not sys_dir.is_dir():
        return False
    for entry in sys_dir.glob("video*"):
        if (entry / "device_caps").is_file() or (entry / "capabilities").is_file():
            return True
    return False


def _linux_heuristic_capture_indices(all_indices: list[int]) -> list[int]:
    """
    Tanpa device_caps di sysfs: UVC ganda biasanya capture di index genap.

    Contoh 2 kamera USB: /dev/video0,1,2,3 -> pakai 0 dan 2 (bukan 1 dan 3).
    """
    if not all_indices:
        return []

    evens = [i for i in all_indices if i % 2 == 0]
    # Pola klasik dua kamera: 0,1,2,3
    if len(all_indices) >= 4 and len(evens) >= 2:
        return evens
    if len(all_indices) == 2 and evens:
        return evens
    return all_indices


def _linux_device_name(index: int) -> str:
    name_file = Path(f"/sys/class/video4linux/video{index}/name")
    if name_file.is_file():
        try:
            return name_file.read_text().strip()
        except OSError:
            pass
    return f"video{index}"


def is_v4l2_capture_node(index: int) -> bool:
    """
    True jika node ini bisa capture video (bukan metadata saja).

    Di Linux, satu kamera USB sering punya /dev/video0 (capture) dan
    /dev/video1 (metadata) — membuka video1 menghasilkan warning OpenCV.
    """
    if not is_linux():
        return True

    base = Path(f"/sys/class/video4linux/video{index}")
    if not base.exists():
        return False

    caps = _read_sysfs_caps(index)
    if caps is None:
        if not _linux_sysfs_has_device_caps():
            all_idx = _linux_video_indices(MAX_CAMERA_PROBE)
            guessed = _linux_heuristic_capture_indices(all_idx)
            if guessed:
                return index in guessed
        return _device_path(index) is not None

    has_capture = bool(caps & _V4L2_CAP_VIDEO_CAPTURE)
    meta_only = bool(caps & _V4L2_CAP_META_CAPTURE) and not has_capture
    return has_capture and not meta_only


def _linux_capture_indices(max_probe: int) -> list[int]:
    """Index /dev/video* yang dipakai untuk capture (sysfs atau heuristik genap)."""
    all_idx = _linux_video_indices(max_probe)
    if not all_idx:
        return []

    if not _linux_sysfs_has_device_caps():
        guessed = _linux_heuristic_capture_indices(all_idx)
        if guessed:
            return guessed

    indices: list[int] = []
    sys_dir = Path("/sys/class/video4linux")
    if not sys_dir.is_dir():
        return _linux_heuristic_capture_indices(all_idx)

    for entry in sorted(sys_dir.glob("video*")):
        suffix = entry.name[5:]
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        if idx >= max_probe:
            continue
        if not _device_path(idx):
            continue
        if is_v4l2_capture_node(idx):
            indices.append(idx)

    return sorted(set(indices)) or _linux_heuristic_capture_indices(all_idx)


def _linux_video_indices(max_probe: int) -> list[int]:
    """Semua /dev/video* (termasuk metadata) — fallback jika sysfs kosong."""
    indices: list[int] = []
    for path in sorted(Path("/dev").glob("video*")):
        suffix = path.name[5:]
        if suffix.isdigit():
            idx = int(suffix)
            if idx < max_probe:
                indices.append(idx)
    return sorted(set(indices))


def _log_linux_capture_nodes(candidates: list[int]) -> None:
    if not is_linux() or not candidates:
        return
    parts = [
        f"/dev/video{i} ({_linux_device_name(i)})" for i in candidates
    ]
    print(f"[CAMERA] Node capture V4L2: {', '.join(parts)}")


def _open_video_capture_impl(index: int) -> cv2.VideoCapture:
    """Buka satu device (tanpa timeout). Linux: satu path /dev/videoN saja."""
    backend = capture_backend()
    if is_linux():
        path = _device_path(index)
        source = path if path else index
        return cv2.VideoCapture(source, backend)

    path = _device_path(index)
    if path is not None:
        return cv2.VideoCapture(path, backend)
    return cv2.VideoCapture(index, backend)


def open_video_capture(index: int) -> cv2.VideoCapture:
    """
    Buka kamera. Di Linux pakai timeout agar tidak hang menit-an
    saat USB sibuk atau device tidak bisa dibuka.
    """
    timeout = CAMERA_OPEN_TIMEOUT_SEC if is_linux() else 0.0
    if timeout <= 0:
        return _open_video_capture_impl(index)

    result: list[cv2.VideoCapture] = []

    def _worker() -> None:
        with _probe_lock:
            try:
                result.append(_open_video_capture_impl(index))
            except Exception:
                result.append(cv2.VideoCapture())

    label = _device_path(index) or f"index {index}"
    print(f"[CAMERA] Membuka {label} (max {timeout:.0f}s)...", flush=True)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(
            f"[CAMERA] Timeout {label} — lewati (atur .env atau cabut colokan lain)",
            flush=True,
        )
        return cv2.VideoCapture()

    cap = result[0] if result else cv2.VideoCapture()
    if cap.isOpened():
        print(f"[CAMERA] OK: {label}", flush=True)
    return cap


def _read_test_frame(cap: cv2.VideoCapture) -> bool:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True
        time.sleep(0.08)
    return False


def _probe_order(max_probe: int) -> list[int]:
    """
  Urutan scan kamera.

  Linux: hanya node capture (biasanya video0 + video2 untuk 2 kamera USB).
  """
    if is_linux():
        capture = _linux_capture_indices(max_probe)
        if capture:
            _log_linux_capture_nodes(capture)
            return capture

        # Fallback jika sysfs tidak tersedia
        on_disk = _linux_video_indices(max_probe)
        return on_disk if on_disk else list(range(max_probe))

    return list(range(max_probe))


def probe_camera(index: int) -> bool:
    """Cek apakah index kamera bisa dibuka dan membaca frame."""
    if is_linux() and not is_v4l2_capture_node(index):
        return False

    retries = 1 if is_linux() and CAMERA_OPEN_TIMEOUT_SEC > 0 else _PROBE_RETRIES
    for _attempt in range(retries):
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

    _probe_settle_delay()

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


def _find_dual_pair(
    found: list[int],
    candidates: list[int],
    max_probe: int,
) -> list[int]:
    """Cari dua index yang bisa dibuka bersamaan."""
    found_set = set(found)
    if len(found_set) >= 2:
        return sorted(found_set)

    # Sudah satu kamera: coba pasangan sekali saja (hindari hang berulang)
    if len(found_set) == 1 and is_linux():
        only = next(iter(found_set))
        pool_one = [i for i in candidates if i != only]
        for other in pool_one:
            print(
                f"[CAMERA] Uji dual {only}+{other}...",
                flush=True,
            )
            if _probe_pair_simultaneous(only, other):
                return sorted({only, other})
        return sorted(found_set)

    # Pasangan yang sudah ketemu satu sisi
    pool = sorted(set(candidates) | found_set)
    if not pool and is_linux():
        pool = _linux_capture_indices(max_probe) or _linux_video_indices(max_probe)
    if not pool:
        pool = list(range(max_probe))

    # Utamakan (0, 2) — pola 2 kamera USB di Linux Mint tanpa device_caps
    preferred: list[tuple[int, int]] = []
    if is_linux() and 0 in pool and 2 in pool:
        preferred.append((0, 2))
    if len(pool) >= 2:
        preferred.append((pool[0], pool[1]))
    for pair in combinations(pool, 2):
        if pair not in preferred:
            preferred.append(pair)

    for index_a, index_b in preferred:
        if index_a in found_set and index_b in found_set:
            return sorted(found_set)
        if _probe_pair_simultaneous(index_a, index_b):
            return sorted(found_set | {index_a, index_b})

    return sorted(found_set)


def list_available_cameras(
    max_probe: int | None = None,
    held_indices: list[int] | None = None,
) -> list[int]:
    """
    Scan index kamera yang aktif.

    held_indices: index yang sedang dipakai — jangan di-probe ulang
    (membuka device yang sama saat sudah terbuka sering gagal).
    """
    _suppress_opencv_probe_noise()

    if max_probe is None:
        max_probe = MAX_CAMERA_PROBE

    held = set(held_indices or [])
    order = _probe_order(max_probe)
    found: list[int] = []

    for index in order:
        if index in held:
            found.append(index)
            continue
        if probe_camera(index):
            found.append(index)
        _probe_settle_delay()

    found = sorted(set(found))
    result = _find_dual_pair(found, order, max_probe)

    if is_linux() and len(result) < 2:
        all_nodes = _linux_video_indices(max_probe)
        capture_nodes = _linux_capture_indices(max_probe)
        print(
            f"[CAMERA] Hanya {len(result)} kamera aktif. "
            f"/dev/video*: {all_nodes} | dicoba capture: {capture_nodes}"
        )
        if not _linux_sysfs_has_device_caps():
            print(
                "[CAMERA] Kernel tanpa device_caps — pakai index genap "
                f"({capture_nodes}). Set .env: CAMERA_NGINTIL=0 CAMERA_BRONDOL=2"
            )
        elif len(capture_nodes) >= 2:
            print(
                "[CAMERA] Tip: set .env CAMERA_NGINTIL="
                f"{capture_nodes[0]} CAMERA_BRONDOL={capture_nodes[1]}"
            )

    return result


def get_fixed_camera_indices() -> list[int] | None:
    """Index dari .env jika NGINTIL dan BRONDOL keduanya diset (skip scan agresif)."""
    ng = _env_camera_index("ngintil")
    br = _env_camera_index("brondol")
    if ng is not None and br is not None and ng != br:
        return [ng, br]
    return None


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


def _assign_by_linux_device_name(
    available: list[int],
) -> tuple[int | None, int | None]:
    """Cocokkan nama sysfs ke role (LUMINOUS=ngintil, 1080P=brondol)."""
    ngintil_index: int | None = None
    brondol_index: int | None = None
    for idx in available:
        name = _linux_device_name(idx).lower()
        if any(k in name for k in ("luminous", "c50")):
            ngintil_index = idx
        if any(k in name for k in ("1080", "web cam", "webcam")):
            brondol_index = idx
    return ngintil_index, brondol_index


def assign_camera_indices(
    available: list[int] | None = None,
) -> tuple[int | None, int | None]:
    """
    Tetapkan index untuk ngintil (pertama) dan brondol (kedua).

    Linux: pakai urutan node capture yang terbukti bisa dibaca
    (mis. video0 + video2), bukan selalu 0 dan 1.
    """
    if available is None:
        available = list_available_cameras()

    ngintil_override = _env_camera_index("ngintil")
    brondol_override = _env_camera_index("brondol")

    ngintil_index: int | None = None
    brondol_index: int | None = None

    if is_linux() and not ngintil_override and not brondol_override:
        ng_by_name, br_by_name = _assign_by_linux_device_name(available)
        if ng_by_name is not None:
            ngintil_index = ng_by_name
        if br_by_name is not None:
            brondol_index = br_by_name

    if ngintil_override is not None:
        ngintil_index = ngintil_override
    elif ngintil_index is None and available:
        ngintil_index = available[0]

    if brondol_override is not None:
        brondol_index = brondol_override
    elif brondol_index is None and len(available) >= 2:
        brondol_index = available[1]

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
