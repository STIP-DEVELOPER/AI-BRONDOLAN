"""Komponen UI dashboard OpenCV — SAVIRA AI."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# --- Tema warna (BGR) ---
COLOR_BG = (24, 26, 30)
COLOR_SURFACE = (32, 36, 42)
COLOR_TEXT = (245, 247, 250)
COLOR_TEXT_DIM = (165, 172, 182)
COLOR_ACCENT = (72, 180, 110)
COLOR_NGINTIL = (60, 140, 255)
COLOR_BRONDOL = (100, 220, 140)
COLOR_DANGER = (80, 90, 240)
COLOR_MUTE_BTN = (48, 52, 62)
COLOR_UNMUTE_BTN = (42, 95, 52)
COLOR_SERIAL_ON = (80, 200, 120)
COLOR_SERIAL_OFF = (90, 100, 190)
COLOR_BORDER = (140, 140, 140)

APP_TITLE = "SAVIRA AI"
HEADER_HEIGHT = 40
MODULE_HEADER_HEIGHT = 36
GAP = 4
BORDER_W = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class DashboardLayout:
    width: int
    height: int
    panel_width: int
    camera_height: int
    voice_height: int
    camera_y: int
    voice_y: int


def _macos_screen_size() -> tuple[int, int] | None:
    """Deteksi resolusi macOS tanpa tkinter (aman dengan OpenCV)."""
    import ctypes
    import ctypes.util
    import re
    import subprocess

    lib_path = ctypes.util.find_library("CoreGraphics")
    if not lib_path:
        lib_path = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
    try:
        class CGPoint(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

        class CGSize(ctypes.Structure):
            _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

        class CGRect(ctypes.Structure):
            _fields_ = [("origin", CGPoint), ("size", CGSize)]

        cg = ctypes.CDLL(lib_path)
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
        cg.CGDisplayBounds.restype = CGRect
        bounds = cg.CGDisplayBounds(cg.CGMainDisplayID())
        w, h = int(bounds.size.width), int(bounds.size.height)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        match = re.search(r"Resolution:\s*(\d+)\s*x\s*(\d+)", out)
        if match:
            w, h = int(match.group(1)), int(match.group(2))
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass

    return None


def _windows_screen_size() -> tuple[int, int] | None:
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        w = int(user32.GetSystemMetrics(0))
        h = int(user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        return None
    return None


def get_primary_screen_size() -> tuple[int, int]:
    """Resolusi layar utama tanpa tkinter (hindari crash macOS + OpenCV)."""
    env_w = os.getenv("UI_SCREEN_WIDTH", "").strip()
    env_h = os.getenv("UI_SCREEN_HEIGHT", "").strip()
    if env_w and env_h:
        try:
            return int(env_w), int(env_h)
        except ValueError:
            pass

    if sys.platform == "darwin":
        size = _macos_screen_size()
        if size:
            return size
    elif sys.platform == "win32":
        size = _windows_screen_size()
        if size:
            return size

    return 1920, 1080


def query_window_size(
    window_name: str,
    fallback: tuple[int, int] | None = None,
    *,
    fullscreen: bool = True,
) -> tuple[int, int]:
    """Ukuran area tampilan jendela OpenCV."""
    default = fallback or get_primary_screen_size()

    try:
        rect = cv2.getWindowImageRect(window_name)
        if rect and len(rect) >= 4:
            w, h = int(rect[2]), int(rect[3])
            if w > 100 and h > 100:
                return w, h
    except cv2.error:
        pass

    return default


def compute_layout(screen_width: int, screen_height: int) -> DashboardLayout:
    """Header menempel atas; kamera langsung di bawah header; suara di bawah."""
    w = max(640, screen_width)
    h = max(480, screen_height)

    voice_h = max(96, min(int(h * 0.14), 150))
    camera_y = HEADER_HEIGHT
    voice_y = h - voice_h
    camera_h = voice_y - camera_y - GAP

    if camera_h < 160:
        camera_h = 160
        voice_y = min(h - 88, camera_y + camera_h + GAP)
        voice_h = h - voice_y

    return DashboardLayout(
        width=w,
        height=h,
        panel_width=w // 2,
        camera_height=camera_h,
        voice_height=voice_h,
        camera_y=camera_y,
        voice_y=voice_y,
    )


def setup_opencv_window(
    window_name: str,
    *,
    width: int | None = None,
    height: int | None = None,
    fullscreen: bool = True,
) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        sw, sh = get_primary_screen_size()
        boot = np.full((sh, sw, 3), COLOR_BG, dtype=np.uint8)
        cv2.imshow(window_name, boot)
        cv2.waitKey(1)
    elif width is not None and height is not None:
        cv2.resizeWindow(window_name, width, height)


def fit_frame(frame: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if frame is None or frame.size == 0:
        placeholder = np.full((height, width, 3), COLOR_SURFACE, dtype=np.uint8)
        cv2.putText(
            placeholder,
            "No signal",
            (16, 36),
            FONT,
            0.6,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )
        return placeholder
    return cv2.resize(frame, (width, height))


def _stroke_rect(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    thickness: int = BORDER_W,
) -> None:
    cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BORDER, thickness, cv2.LINE_AA)


def _hline(canvas: np.ndarray, y: int) -> None:
    cv2.line(
        canvas,
        (0, y),
        (canvas.shape[1] - 1, y),
        COLOR_BORDER,
        BORDER_W,
        cv2.LINE_AA,
    )


def _vline(canvas: np.ndarray, x: int, y1: int, y2: int) -> None:
    cv2.line(canvas, (x, y1), (x, y2), COLOR_BORDER, BORDER_W, cv2.LINE_AA)


def _draw_module_header(canvas: np.ndarray, module_label: str) -> None:
    bar = canvas[0:MODULE_HEADER_HEIGHT, :]
    bar[:] = (20, 38, 32)
    label = f"{APP_TITLE}  |  {module_label}"
    scale = 0.6
    th = 2
    _, th_px = cv2.getTextSize(label, FONT, scale, th)
    text_y = (MODULE_HEADER_HEIGHT + th_px) // 2
    cv2.putText(bar, label, (12, text_y), FONT, scale, COLOR_TEXT, th, cv2.LINE_AA)
    _hline(canvas, MODULE_HEADER_HEIGHT - 1)


def wrap_text(text: str, max_chars: int = 52) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_button(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    fill: tuple[int, int, int],
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
    ts, _ = cv2.getTextSize(label, FONT, 0.55, 2)
    tx = x1 + ((x2 - x1) - ts[0]) // 2
    ty = y1 + ((y2 - y1) + ts[1]) // 2
    cv2.putText(img, label, (tx, ty), FONT, 0.55, COLOR_TEXT, 2, cv2.LINE_AA)


def draw_top_bar(
    canvas: np.ndarray,
    *,
    serial_connected: bool,
    muted: bool,
) -> tuple[int, int, int, int]:
    bar = canvas[0:HEADER_HEIGHT, :]
    bar[:] = (20, 38, 32)

    title_scale = 0.78
    title_th = 2
    _, title_h = cv2.getTextSize(APP_TITLE, FONT, title_scale, title_th)
    title_y = (HEADER_HEIGHT + title_h) // 2
    cv2.putText(
        bar,
        APP_TITLE,
        (12, title_y),
        FONT,
        title_scale,
        COLOR_TEXT,
        title_th,
        cv2.LINE_AA,
    )

    serial_color = COLOR_SERIAL_ON if serial_connected else COLOR_SERIAL_OFF
    serial_cy = HEADER_HEIGHT // 2
    cv2.circle(bar, (bar.shape[1] - 188, serial_cy), 7, serial_color, -1, cv2.LINE_AA)

    btn_w, btn_h = 96, 28
    btn_x2 = bar.shape[1] - 10
    btn_x1 = btn_x2 - btn_w
    btn_y1 = (HEADER_HEIGHT - btn_h) // 2
    btn_y2 = btn_y1 + btn_h

    if muted:
        _draw_button(bar, btn_x1, btn_y1, btn_x2, btn_y2, "UNMUTE", COLOR_UNMUTE_BTN)
    else:
        _draw_button(bar, btn_x1, btn_y1, btn_x2, btn_y2, "MUTE", COLOR_MUTE_BTN)

    return (btn_x1, btn_y1, btn_x2, btn_y2)


def decorate_camera_view(
    frame: np.ndarray,
    title: str,
    command: str | None,
    accent: tuple[int, int, int],
    object_count: int | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    cv2.putText(out, title, (10, 24), FONT, 0.58, COLOR_TEXT, 2, cv2.LINE_AA)

    if object_count is not None and object_count > 0:
        cv2.putText(
            out,
            str(object_count),
            (w - 32, 24),
            FONT,
            0.52,
            accent,
            2,
            cv2.LINE_AA,
        )

    badge = command if command else "—"
    badge_scale = 0.55
    ts, _ = cv2.getTextSize(badge, FONT, badge_scale, 2)
    bx2 = w - 8
    bx1 = max(8, bx2 - ts[0] - 14)
    by1, by2 = h - 38, h - 8
    cv2.rectangle(out, (bx1, by1), (bx2, by2), accent, -1, cv2.LINE_AA)
    cv2.putText(
        out,
        badge,
        (bx1 + 7, by2 - 9),
        FONT,
        badge_scale,
        (18, 18, 18),
        2,
        cv2.LINE_AA,
    )

    return out


def draw_camera_tile(
    frame: np.ndarray | None,
    width: int,
    height: int,
    title: str,
    command: str | None,
    accent: tuple[int, int, int],
    object_count: int | None = None,
) -> np.ndarray:
    content = fit_frame(frame, width, height)
    return decorate_camera_view(content, title, command, accent, object_count)


def draw_voice_panel(
    canvas: np.ndarray,
    y_start: int,
    height: int,
    status: str,
    user_text: str,
    bot_text: str,
    volume: float = 0.0,
    muted: bool = False,
) -> None:
    width = canvas.shape[1]
    panel = canvas[y_start : y_start + height, :]
    panel[:] = COLOR_SURFACE

    margin = 12
    bar_y = 10
    bar_h = 10
    bar_w = width - margin * 2
    cv2.rectangle(
        panel,
        (margin, bar_y),
        (margin + bar_w, bar_y + bar_h),
        (45, 48, 54),
        -1,
    )
    fill_w = int(min(volume * 9000, bar_w))
    bar_color = COLOR_DANGER if muted else COLOR_ACCENT
    if fill_w > 0:
        cv2.rectangle(
            panel,
            (margin, bar_y),
            (margin + fill_w, bar_y + bar_h),
            bar_color,
            -1,
        )

    status_color = COLOR_DANGER if muted else COLOR_TEXT_DIM
    max_chars = max(28, (width - margin * 2) // 8)
    status_line = (status or "Siap")[:max_chars]
    cv2.putText(
        panel,
        status_line,
        (margin, 38),
        FONT,
        0.45,
        status_color,
        1,
        cv2.LINE_AA,
    )

    y = 52
    line_h = 20
    max_lines = max(1, (height - 56) // line_h // 2)
    for text, color in ((user_text, (190, 200, 255)), (bot_text, COLOR_BRONDOL)):
        if not text:
            continue
        for line in wrap_text(text, max_chars=max_chars)[:max_lines]:
            if y + line_h > height - 6:
                break
            cv2.putText(panel, line, (margin, y), FONT, 0.46, color, 1, cv2.LINE_AA)
            y += line_h
        y += 4


def build_dashboard(
    ngintil_frame: np.ndarray | None,
    brondol_frame: np.ndarray | None,
    ngintil_command: str | None,
    brondol_command: str | None,
    brondol_object_count: int,
    voice_state: dict,
    *,
    screen_width: int,
    screen_height: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    layout = compute_layout(screen_width, screen_height)

    canvas = np.full((layout.height, layout.width, 3), COLOR_BG, dtype=np.uint8)

    mute_rect = draw_top_bar(
        canvas,
        serial_connected=voice_state.get("serial_connected", False),
        muted=voice_state.get("muted", False),
    )

    gap_mid = GAP // 2 or 2
    left_w = layout.panel_width - gap_mid
    right_w = layout.width - layout.panel_width - gap_mid
    left_x = 0
    right_x = layout.panel_width + gap_mid

    left = draw_camera_tile(
        ngintil_frame,
        left_w,
        layout.camera_height,
        "NGINTIL",
        ngintil_command,
        COLOR_NGINTIL,
    )
    right = draw_camera_tile(
        brondol_frame,
        right_w,
        layout.camera_height,
        "BRONDOLAN",
        brondol_command,
        COLOR_BRONDOL,
        object_count=brondol_object_count,
    )

    cy = layout.camera_y
    ch = layout.camera_height
    canvas[cy : cy + ch, left_x : left_x + left_w] = left
    canvas[cy : cy + ch, right_x : right_x + right_w] = right

    draw_voice_panel(
        canvas,
        y_start=layout.voice_y,
        height=layout.height - layout.voice_y,
        status=voice_state.get("status", ""),
        user_text=voice_state.get("user_text", ""),
        bot_text=voice_state.get("bot_text", ""),
        volume=voice_state.get("volume", 0.0),
        muted=voice_state.get("muted", False),
    )

    w, h = layout.width, layout.height
    cy, ch = layout.camera_y, layout.camera_height
    _hline(canvas, HEADER_HEIGHT - 1)
    _stroke_rect(canvas, 0, 0, w - 1, HEADER_HEIGHT - 1)
    _hline(canvas, layout.voice_y - 1)
    _stroke_rect(canvas, 0, layout.voice_y, w - 1, h - 1)
    _vline(canvas, layout.panel_width, cy, cy + ch - 1)
    _stroke_rect(canvas, 0, cy, left_x + left_w - 1, cy + ch - 1)
    _stroke_rect(canvas, right_x, cy, right_x + right_w - 1, cy + ch - 1)

    return canvas, mute_rect


def build_voice_dashboard(
    voice_state: dict,
    *,
    screen_width: int,
    screen_height: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Layout standalone voice (run_voice.py)."""
    w = max(640, screen_width)
    h = max(480, screen_height)

    canvas = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
    mute_rect = draw_top_bar(
        canvas,
        serial_connected=voice_state.get("serial_connected", False),
        muted=voice_state.get("muted", False),
    )

    voice_y = HEADER_HEIGHT
    draw_voice_panel(
        canvas,
        y_start=voice_y,
        height=h - voice_y,
        status=voice_state.get("status", ""),
        user_text=voice_state.get("user_text", ""),
        bot_text=voice_state.get("bot_text", ""),
        volume=voice_state.get("volume", 0.0),
        muted=voice_state.get("muted", False),
    )

    _hline(canvas, HEADER_HEIGHT - 1)
    _stroke_rect(canvas, 0, 0, w - 1, HEADER_HEIGHT - 1)
    _hline(canvas, voice_y - 1)
    _stroke_rect(canvas, 0, voice_y, w - 1, h - 1)

    return canvas, mute_rect


def build_camera_module_view(
    frame: np.ndarray,
    module_label: str,
    command: str | None,
    accent: tuple[int, int, int],
    *,
    screen_width: int,
    screen_height: int,
    object_count: int | None = None,
) -> np.ndarray:
    """Layout standalone kamera (run_ngintil.py / run_brondol.py)."""
    w = max(640, screen_width)
    h = max(480, screen_height)

    canvas = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
    _draw_module_header(canvas, module_label)

    cam_y = MODULE_HEADER_HEIGHT
    cam_h = h - cam_y - GAP
    tile = draw_camera_tile(
        frame,
        w - GAP * 2,
        cam_h,
        module_label,
        command,
        accent,
        object_count=object_count,
    )
    x1 = GAP
    canvas[cam_y : cam_y + cam_h, x1 : x1 + tile.shape[1]] = tile

    _hline(canvas, MODULE_HEADER_HEIGHT - 1)
    _stroke_rect(canvas, 0, 0, w - 1, MODULE_HEADER_HEIGHT - 1)
    _hline(canvas, cam_y - 1)
    _stroke_rect(canvas, x1, cam_y, x1 + tile.shape[1] - 1, cam_y + cam_h - 1)

    return canvas


def draw_panel_header(frame: np.ndarray, title: str) -> None:
    decorated = decorate_camera_view(frame, title, None, COLOR_ACCENT)
    frame[:] = decorated
