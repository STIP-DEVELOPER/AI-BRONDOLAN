import sys
import time
from typing import Literal

import cv2
import numpy as np

from src.config.performance import (
    CAMERA_HEIGHT,
    CAMERA_PROBE_DELAY_SEC,
    CAMERA_SCAN_INTERVAL_CONNECTED_SEC,
    CAMERA_SCAN_INTERVAL_SEC,
    CAMERA_WIDTH,
)
from src.services.camera_autodetect import (
    assign_camera_indices,
    is_linux,
    list_available_cameras,
    open_video_capture,
)

CameraRole = Literal["ngintil", "brondol"]

FAIL_READS_BEFORE_RESCAN = 12


def make_camera_off_frame(
    width: int = 640,
    height: int = 480,
    role_label: str = "Kamera",
) -> np.ndarray:
    frame = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        frame,
        f"{role_label} — DIMATIKAN",
        (20, height // 2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (80, 90, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Hemat CPU — klik HIDUPKAN di kanan atas",
        (20, height // 2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (120, 130, 150),
        1,
        cv2.LINE_AA,
    )
    return frame


def make_no_signal_frame(
    width: int = 640,
    height: int = 480,
    message: str = "Kamera tidak terdeteksi",
) -> np.ndarray:
    frame = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (120, 130, 140),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Colokkan kamera USB",
        (20, height // 2 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (90, 100, 110),
        1,
        cv2.LINE_AA,
    )
    return frame


class CameraManager:
    """
    Kelola satu atau dua kamera dengan autodetect & hot-plug.
    Role 'ngintil' -> kamera pertama, 'brondol' -> kamera kedua (jika ada).
    """

    def __init__(self, roles: list[CameraRole] | None = None):
        self._roles: set[CameraRole] = set(roles or ("ngintil", "brondol"))
        self._caps: dict[CameraRole, cv2.VideoCapture | None] = {
            "ngintil": None,
            "brondol": None,
        }
        self._indices: dict[CameraRole, int | None] = {
            "ngintil": None,
            "brondol": None,
        }
        self._available: list[int] = []
        self._last_scan = 0.0
        self._fail_counts: dict[CameraRole, int] = {"ngintil": 0, "brondol": 0}
        self._enabled: dict[CameraRole, bool] = {
            "ngintil": True,
            "brondol": True,
        }

        self.refresh(force=True)

    def _open_capture(self, index: int) -> cv2.VideoCapture | None:
        cap = open_video_capture(index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            time.sleep(0.05)
        cap.release()
        return None

    def _release_role(self, role: CameraRole) -> None:
        cap = self._caps.get(role)
        if cap is not None:
            cap.release()
        self._caps[role] = None
        self._indices[role] = None
        self._fail_counts[role] = 0

    def _release_all(self) -> None:
        for role in list(self._roles):
            self._release_role(role)

    def _scan_interval(self) -> float:
        if self._all_enabled_connected():
            return CAMERA_SCAN_INTERVAL_CONNECTED_SEC
        return CAMERA_SCAN_INTERVAL_SEC

    def _all_enabled_connected(self) -> bool:
        for role in self._roles:
            if not self.is_enabled(role):
                continue
            cap = self._caps.get(role)
            if cap is None or not cap.isOpened():
                return False
        return bool(self._roles)

    def refresh(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_scan < self._scan_interval():
            return

        self._last_scan = now
        held = [
            idx
            for role in self._roles
            if (idx := self._indices.get(role)) is not None
            and self._caps.get(role) is not None
            and self._caps[role].isOpened()
        ]
        new_available = list_available_cameras(held_indices=held)
        if new_available != self._available or force:
            print(f"[CAMERA] Terdeteksi index: {new_available or 'tidak ada'}")
        ng_idx, br_idx = assign_camera_indices(new_available)

        # Mode standalone: satu role + satu kamera fisik -> pakai kamera itu
        if self._roles == {"ngintil"} and new_available:
            ng_idx = ng_idx if ng_idx is not None else new_available[0]
            br_idx = None
        elif self._roles == {"brondol"} and new_available:
            br_idx = br_idx if br_idx is not None else new_available[0]
            ng_idx = None

        target: dict[CameraRole, int | None] = {
            "ngintil": ng_idx if "ngintil" in self._roles else None,
            "brondol": br_idx if "brondol" in self._roles else None,
        }

        changed = new_available != self._available or any(
            self._indices[r] != target[r] for r in self._roles
        )
        self._available = new_available

        if not changed and not force:
            return

        for role in ("ngintil", "brondol"):
            if role not in self._roles:
                continue
            if not self.is_enabled(role):
                if self._indices.get(role) is not None:
                    self._release_role(role)
                continue

            desired = target[role]
            current = self._indices.get(role)

            if desired is None:
                if current is not None:
                    print(f"[CAMERA] {role.upper()}: tidak ada kamera — tunggu colokan")
                    self._release_role(role)
                continue

            if (
                current == desired
                and self._caps[role] is not None
                and self._caps[role].isOpened()
            ):
                continue

            self._release_role(role)
            if (sys.platform == "darwin" or is_linux()) and self._caps.get(
                "ngintil"
            ) is not None:
                time.sleep(CAMERA_PROBE_DELAY_SEC)

            cap = self._open_capture(desired)
            if cap is not None:
                self._caps[role] = cap
                self._indices[role] = desired
                path_hint = f" (/dev/video{desired})" if is_linux() else ""
                print(f"[CAMERA] {role.upper()}: terhubung ke index {desired}{path_hint}")
            else:
                print(f"[CAMERA] {role.upper()}: gagal buka index {desired}")
                if sys.platform == "darwin" or is_linux():
                    time.sleep(CAMERA_PROBE_DELAY_SEC)
                    cap = self._open_capture(desired)
                    if cap is not None:
                        self._caps[role] = cap
                        self._indices[role] = desired
                        print(
                            f"[CAMERA] {role.upper()}: terhubung ke index {desired} (retry)"
                        )

    def is_enabled(self, role: CameraRole) -> bool:
        return self._enabled.get(role, True)

    def set_enabled(self, role: CameraRole, enabled: bool) -> None:
        if self.is_enabled(role) == enabled:
            return

        self._enabled[role] = enabled
        if enabled:
            print(f"[CAMERA] {role.upper()}: dinyalakan")
            self.refresh(force=True)
        else:
            self._release_role(role)
            print(f"[CAMERA] {role.upper()}: dimatikan")

    def toggle_enabled(self, role: CameraRole) -> bool:
        enabled = not self.is_enabled(role)
        self.set_enabled(role, enabled)
        return enabled

    def read(self, role: CameraRole) -> tuple[bool, np.ndarray | None]:
        if not self.is_enabled(role):
            return False, None

        self.refresh()

        cap = self._caps.get(role)
        if cap is None or not cap.isOpened():
            return False, None

        ret, frame = cap.read()
        if not ret or frame is None:
            self._fail_counts[role] += 1
            if self._fail_counts[role] >= FAIL_READS_BEFORE_RESCAN:
                print(f"[CAMERA] {role.upper()}: sinyal hilang — scan ulang...")
                self._release_role(role)
                self.refresh(force=True)
            return False, None

        self._fail_counts[role] = 0
        return True, frame

    def is_connected(self, role: CameraRole) -> bool:
        return self._caps.get(role) is not None and self._indices.get(role) is not None

    def get_index(self, role: CameraRole) -> int | None:
        return self._indices.get(role)

    def get_status_text(self) -> str:
        parts = []
        for role in ("ngintil", "brondol"):
            if role not in self._roles:
                continue
            if not self.is_enabled(role):
                parts.append(f"{role}=DISABLED")
            elif self._indices.get(role) is not None:
                parts.append(f"{role}={self._indices[role]}")
            else:
                parts.append(f"{role}=OFF")
        return " | ".join(parts)

    def close(self) -> None:
        self._release_all()
