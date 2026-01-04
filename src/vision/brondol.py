from ultralytics import YOLO


class BrondolDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def infer(self, frame):
        return self.model(
            frame,
            imgsz=640,
            conf=self.conf,
            device="cpu",
            verbose=False,
        )

    def get_command(self, results, frame_shape):
        if results.boxes is None or len(results.boxes) == 0:
            return None

        box = max(results.boxes, key=lambda b: float(b.conf[0]))

        return "TAKE"
