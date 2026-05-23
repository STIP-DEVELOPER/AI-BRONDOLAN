import time

import cv2
from dotenv import load_dotenv
from src.vision.brondol import BrondolDetector

from src.config.performance import APP_FULLSCREEN, DISPLAY_WAIT_MS
from src.services.DetectionCsvService import DetectionCsvService
from src.services.camera_manager import (
    CameraManager,
    make_camera_off_frame,
    make_no_signal_frame,
)
from src.services.serial_manager import get_shared_serial, release_shared_serial
from src.ui.camera_controls import CameraToggleUI
from src.ui.cv2_dashboard import (
    COLOR_BRONDOL,
    build_camera_module_view,
    query_window_size,
    setup_opencv_window,
)
from src.vision.inference_runner import InferenceRunner

load_dotenv()

MODEL_PATH_BRONDOL = "src/models/brondol_ncnn_model"
ROLE = "brondol"
WINDOW_NAME = "SAVIRA AI — Brondolan"


def main() -> None:
    brondol_detector = BrondolDetector(MODEL_PATH_BRONDOL)
    serial_service = get_shared_serial()
    camera = CameraManager(roles=[ROLE])
    csv_log_service = DetectionCsvService(file_path="logs/detections.csv")
    runner = InferenceRunner()

    setup_opencv_window(WINDOW_NAME, fullscreen=APP_FULLSCREEN)
    screen_w, screen_h = query_window_size(WINDOW_NAME, fullscreen=APP_FULLSCREEN)
    camera_ui = CameraToggleUI(WINDOW_NAME, camera, ROLE, "BRONDOLAN")

    print("Brondolan mode ringan — MATIKAN kamera saat tidak dipakai (hemat CPU)")
    print(f"[CAMERA] {camera.get_status_text()}")

    last_command = None
    brondol_object_count = 0
    last_log_time = time.time()
    placeholder = make_no_signal_frame(message="BRONDOLAN — kamera tidak terdeteksi")
    off_frame = make_camera_off_frame(role_label="BRONDOLAN")

    try:
        while True:
            if not camera.is_enabled(ROLE):
                display = off_frame.copy()
                brondol_object_count = 0
            else:
                ok, frame = camera.read(ROLE)
                if not ok or frame is None:
                    display = placeholder.copy()
                    brondol_object_count = 0
                else:

                    def on_cmd(cmd: str | None) -> None:
                        nonlocal last_command
                        if cmd != last_command and cmd is not None:
                            print(cmd)
                            if serial_service is not None:
                                serial_service.send(cmd)
                            last_command = cmd

                    display, cmd, results = runner.run(
                        brondol_detector,
                        frame,
                        get_command=brondol_detector.get_command,
                        on_command_change=on_cmd,
                    )
                    if cmd is not None:
                        last_command = cmd

                    if results is not None and results.boxes is not None:
                        brondol_object_count = len(results.boxes)
                        now = time.time()
                        if now - last_log_time >= 1.0 and brondol_object_count > 0:
                            csv_log_service.log_detection(
                                total_objects=brondol_object_count
                            )
                            last_log_time = now
                    else:
                        brondol_object_count = 0

            screen_w, screen_h = query_window_size(
                WINDOW_NAME, (screen_w, screen_h), fullscreen=APP_FULLSCREEN
            )
            display = build_camera_module_view(
                display,
                "BRONDOLAN",
                last_command,
                COLOR_BRONDOL,
                screen_width=screen_w,
                screen_height=screen_h,
                object_count=brondol_object_count,
            )
            display = camera_ui.apply_buttons(display)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
            if key == 27:
                break
            if key in (ord("c"), ord("C")):
                camera.toggle_enabled(ROLE)
                runner.reset()
    finally:
        camera.close()
        release_shared_serial()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
