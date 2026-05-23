from ultralytics import YOLO

from src.config.performance import INFER_CONF, INFER_IMGSZ, INFER_IOU


class NgintilDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float | None = None,
        left_ratio: float = 0.4,
        right_ratio: float = 0.6,
        stop_area_ratio: float = 0.35,
        imgsz: int | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold if conf_threshold is not None else INFER_CONF
        self.imgsz = imgsz if imgsz is not None else INFER_IMGSZ
        self.left_ratio = left_ratio
        self.right_ratio = right_ratio
        self.stop_area_ratio = stop_area_ratio

    def infer(self, frame):
        return self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=INFER_IOU,
            max_det=20,
            device="cpu",
            verbose=False,
        )

    def get_command(self, results, frame_shape):
        # frame_shape dari results.orig_shape (H, W) ultralytics
        h, w = int(frame_shape[0]), int(frame_shape[1])
        frame_area = w * h

        # default fail-safe
        command = "STOP"

        if results.boxes is None:
            return command

        persons = [
            box for box in results.boxes
            if int(box.cls[0]) == 0
        ]

        if not persons:
            return command

        # person dengan confidence tertinggi
        box = max(persons, key=lambda b: float(b.conf[0]))

        x1, y1, x2, y2 = box.xyxy[0]
        bbox_area = (x2 - x1) * (y2 - y1)
        x_center = (x1 + x2) / 2
        x_ratio = x_center / w

        if bbox_area / frame_area > self.stop_area_ratio:
            return "STOP"
        if x_ratio < self.left_ratio:
            return "LEFT"
        if x_ratio > self.right_ratio:
            return "RIGHT"

        return "FORWARD"
