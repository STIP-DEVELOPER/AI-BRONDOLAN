#!/usr/bin/env python3
"""Uji kamera V4L2 di Linux tanpa v4l2-ctl. Jalankan dari root proyek."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.camera_autodetect import (  # noqa: E402
    _linux_capture_indices,
    _linux_device_name,
    _linux_sysfs_has_device_caps,
    _linux_video_indices,
    list_available_cameras,
    probe_camera,
)


def main() -> None:
    print("=== Probe kamera Linux ===\n")
    all_nodes = _linux_video_indices(10)
    print(f"/dev/video*     : {all_nodes}")
    print(f"device_caps sysfs: {_linux_sysfs_has_device_caps()}")
    capture = _linux_capture_indices(10)
    print(f"Dicoba capture  : {capture}")
    for idx in capture:
        print(f"  /dev/video{idx} — {_linux_device_name(idx)}")

    print("\n--- Uji buka + baca frame ---")
    for idx in capture:
        ok = probe_camera(idx)
        print(f"  video{idx}: {'OK' if ok else 'GAGAL'}")

    print("\n--- Autodetect (dual) ---")
    found = list_available_cameras()
    print(f"Terdeteksi index: {found}")
    if found:
        print("\nSaran .env:")
        print(f"  CAMERA_NGINTIL={found[0]}")
        if len(found) > 1:
            print(f"  CAMERA_BRONDOL={found[1]}")


if __name__ == "__main__":
    main()
