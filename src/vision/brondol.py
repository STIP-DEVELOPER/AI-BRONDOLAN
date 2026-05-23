from ultralytics import YOLO

from src.config.performance import INFER_CONF, INFER_IMGSZ, INFER_IOU


class BrondolDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float | None = None,
        imgsz: int | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold if conf_threshold is not None else INFER_CONF
        self.imgsz = imgsz if imgsz is not None else INFER_IMGSZ

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
        if results.boxes is None or len(results.boxes) == 0:
            return None

        box = max(results.boxes, key=lambda b: float(b.conf[0]))

        return "TAKE"
