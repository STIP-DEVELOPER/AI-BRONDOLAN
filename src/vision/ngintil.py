from ultralytics import YOLO


class NgintilDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        left_ratio: float = 0.4,
        right_ratio: float = 0.6,
        stop_area_ratio: float = 0.35,
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.left_ratio = left_ratio
        self.right_ratio = right_ratio
        self.stop_area_ratio = stop_area_ratio

    def infer(self, frame):
        return self.model(
            frame,
            imgsz=640,
            conf=self.conf,
            device="cpu",
            verbose=False,
        )

    def get_command(self, results, frame_shape):
        h, w = frame_shape[:2]
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
