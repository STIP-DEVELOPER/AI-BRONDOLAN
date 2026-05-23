"""Inferensi YOLO + gambar bbox (koordinat benar, tanpa menumpuk)."""

from __future__ import annotations

import cv2
import numpy as np

from src.config.performance import INFER_EVERY_N_FRAMES, MAX_DISPLAY_BOXES

BOX_COLOR = (80, 220, 120)
BOX_THICKNESS = 2


def _scale_xyxy_to_frame(
    xyxy,
    orig_shape: tuple,
    frame_shape: tuple,
) -> tuple[int, int, int, int]:
    """Konversi xyxy dari ruang results.orig_shape ke pixel frame tampilan."""
    oh, ow = int(orig_shape[0]), int(orig_shape[1])
    fh, fw = frame_shape[0], frame_shape[1]

    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if ow > 0 and oh > 0 and (ow != fw or oh != fh):
        x1 = x1 * fw / ow
        x2 = x2 * fw / ow
        y1 = y1 * fh / oh
        y2 = y2 * fh / oh

    x1, x2 = sorted((max(0, min(fw - 1, int(x1))), max(0, min(fw - 1, int(x2)))))
    y1, y2 = sorted((max(0, min(fh - 1, int(y1))), max(0, min(fh - 1, int(y2)))))
    return x1, y1, x2, y2


def _select_boxes(results, max_boxes: int, class_id: int | None) -> list:
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return []

    boxes = list(results.boxes)
    if class_id is not None:
        boxes = [b for b in boxes if int(b.cls[0]) == class_id]

    boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)
    return boxes[:max_boxes]


def annotate_frame(
    frame: np.ndarray,
    results,
    *,
    max_boxes: int | None = None,
    class_id: int | None = None,
) -> np.ndarray:
    if max_boxes is None:
        max_boxes = MAX_DISPLAY_BOXES

    out = frame.copy()
    selected = _select_boxes(results, max_boxes, class_id)
    if not selected:
        return out

    orig_shape = results.orig_shape

    for box in selected:
        x1, y1, x2, y2 = _scale_xyxy_to_frame(
            box.xyxy[0],
            orig_shape,
            out.shape,
        )
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS, cv2.LINE_AA)

        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        cv2.putText(
            out,
            f"{cls_id} {int(conf * 100)}%",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BOX_COLOR,
            1,
            cv2.LINE_AA,
        )

    return out


class InferenceRunner:
    def __init__(self, every_n: int | None = None):
        self.every_n = every_n if every_n is not None else INFER_EVERY_N_FRAMES
        self._tick = 0
        self._last_command: str | None = None
        self._last_results = None

    def run(
        self,
        detector,
        frame: np.ndarray,
        *,
        get_command,
        on_command_change=None,
        infer_this_frame: bool = True,
        max_boxes: int | None = None,
        class_id: int | None = None,
    ) -> tuple[np.ndarray, str | None, object | None]:
        self._tick += 1

        do_infer = infer_this_frame and (
            self._last_results is None or self._tick % self.every_n == 0
        )

        if do_infer:
            results = detector.infer(frame)
            self._last_results = results[0]
            command = get_command(self._last_results, self._last_results.orig_shape)
            self._last_command = command
            if on_command_change and command is not None:
                on_command_change(command)

        display = annotate_frame(
            frame,
            self._last_results,
            max_boxes=max_boxes,
            class_id=class_id,
        )
        return display, self._last_command, self._last_results

    @property
    def has_results(self) -> bool:
        return self._last_results is not None

    @property
    def last_box_count(self) -> int:
        if self._last_results is None or self._last_results.boxes is None:
            return 0
        return len(self._last_results.boxes)

    def reset(self) -> None:
        self._tick = 0
        self._last_command = None
        self._last_results = None
