"""Tombol hidupkan/matikan kamera untuk jendela OpenCV standalone."""

from __future__ import annotations

import cv2
import numpy as np

from src.services.camera_manager import CameraManager, CameraRole

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_BTN_ON = (50, 130, 70)
COLOR_BTN_OFF = (50, 55, 130)
COLOR_BTN_DIM = (42, 45, 50)
COLOR_BTN_DIM_TEXT = (100, 105, 115)
COLOR_BORDER = (255, 255, 255)
COLOR_TEXT = (245, 245, 245)

BUTTON_W = 148
BUTTON_H = 36
BUTTON_GAP = 8
MARGIN = 12


def _draw_btn(
    img: np.ndarray,
    x1: int,
    y1: int,
    label: str,
    fill: tuple[int, int, int],
    active: bool,
) -> tuple[int, int, int, int]:
    x2 = x1 + BUTTON_W
    y2 = y1 + BUTTON_H
    if not active:
        fill = COLOR_BTN_DIM
        text_color = COLOR_BTN_DIM_TEXT
    else:
        text_color = COLOR_TEXT

    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BORDER, 2, cv2.LINE_AA)
    ts, _ = cv2.getTextSize(label, FONT, 0.48, 1)
    tx = x1 + (BUTTON_W - ts[0]) // 2
    ty = y1 + (BUTTON_H + ts[1]) // 2
    cv2.putText(img, label, (tx, ty), FONT, 0.48, text_color, 1, cv2.LINE_AA)
    return (x1, y1, x2, y2)


def draw_camera_power_buttons(
    frame: np.ndarray,
    camera_enabled: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Gambar tombol HIDUPKAN (kiri) dan MATIKAN (kanan) di pojok kanan atas.
    Return (rect_hidupkan, rect_matikan) dalam koordinat frame.
    """
    out = frame
    h, w = out.shape[:2]

    rect_off_x2 = w - MARGIN
    rect_off_x1 = rect_off_x2 - BUTTON_W
    rect_on_x2 = rect_off_x1 - BUTTON_GAP
    rect_on_x1 = rect_on_x2 - BUTTON_W
    y1 = MARGIN
    y2 = y1 + BUTTON_H

    # Bar semi-transparan di belakang tombol
    overlay = out.copy()
    cv2.rectangle(overlay, (rect_on_x1 - 6, 0), (w, y2 + 8), (20, 22, 26), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    rect_on = _draw_btn(
        out,
        rect_on_x1,
        y1,
        "HIDUPKAN",
        COLOR_BTN_ON,
        active=not camera_enabled,
    )
    rect_off = _draw_btn(
        out,
        rect_off_x1,
        y1,
        "MATIKAN",
        COLOR_BTN_OFF,
        active=camera_enabled,
    )

    return rect_on, rect_off


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


class CameraToggleUI:
    """Mouse handler + tombol kamera untuk satu jendela OpenCV."""

    def __init__(
        self,
        window_name: str,
        camera: CameraManager,
        role: CameraRole,
        role_label: str,
    ):
        self.window_name = window_name
        self.camera = camera
        self.role = role
        self.role_label = role_label
        self._rect_on: tuple[int, int, int, int] | None = None
        self._rect_off: tuple[int, int, int, int] | None = None
        cv2.setMouseCallback(window_name, self._on_mouse)

    def apply_buttons(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        self._rect_on, self._rect_off = draw_camera_power_buttons(
            display,
            camera_enabled=self.camera.is_enabled(self.role),
        )
        return display

    def _on_mouse(self, event: int, x: int, y: int, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        enabled = self.camera.is_enabled(self.role)

        if (
            not enabled
            and self._rect_on is not None
            and point_in_rect(x, y, self._rect_on)
        ):
            self.camera.set_enabled(self.role, True)
            print(f"[CAMERA] {self.role_label}: dihidupkan (tombol)")
            return

        if (
            enabled
            and self._rect_off is not None
            and point_in_rect(x, y, self._rect_off)
        ):
            self.camera.set_enabled(self.role, False)
            print(f"[CAMERA] {self.role_label}: dimatikan (tombol — hemat CPU)")
