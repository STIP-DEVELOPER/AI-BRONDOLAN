import cv2
from src.vision.ngintil import NgintilDetector

from src.config.performance import APP_FULLSCREEN, DISPLAY_WAIT_MS
from src.services.camera_manager import (
    CameraManager,
    make_camera_off_frame,
    make_no_signal_frame,
)
from src.services.serial_manager import get_shared_serial, release_shared_serial
from src.ui.camera_controls import CameraToggleUI
from src.ui.cv2_dashboard import (
    COLOR_NGINTIL,
    build_camera_module_view,
    query_window_size,
    setup_opencv_window,
)
from src.vision.inference_runner import InferenceRunner

MODEL_PATH_NGINTIL = "src/models/ngintil_ncnn_model"
ROLE = "ngintil"
WINDOW_NAME = "SAVIRA AI — Ngintil"


def main() -> None:
    ngintil_detector = NgintilDetector(MODEL_PATH_NGINTIL)
    serial_service = get_shared_serial()
    camera = CameraManager(roles=[ROLE])
    runner = InferenceRunner()

    setup_opencv_window(WINDOW_NAME, fullscreen=APP_FULLSCREEN)
    screen_w, screen_h = query_window_size(WINDOW_NAME, fullscreen=APP_FULLSCREEN)
    camera_ui = CameraToggleUI(WINDOW_NAME, camera, ROLE, "NGINTIL")

    print("Ngintil mode ringan — MATIKAN kamera saat tidak dipakai (hemat CPU)")
    print(f"[CAMERA] {camera.get_status_text()}")

    last_command = None
    placeholder = make_no_signal_frame(message="NGINTIL — kamera tidak terdeteksi")
    off_frame = make_camera_off_frame(role_label="NGINTIL")

    try:
        while True:
            if not camera.is_enabled(ROLE):
                display = off_frame.copy()
            else:
                ok, frame = camera.read(ROLE)
                if not ok or frame is None:
                    display = placeholder.copy()
                else:

                    def on_cmd(cmd: str | None) -> None:
                        nonlocal last_command
                        if cmd != last_command and cmd is not None:
                            print(cmd)
                            if serial_service is not None:
                                serial_service.send(cmd)
                            last_command = cmd

                    display, cmd, _ = runner.run(
                        ngintil_detector,
                        frame,
                        get_command=ngintil_detector.get_command,
                        on_command_change=on_cmd,
                        class_id=0,
                    )
                    if cmd is not None:
                        last_command = cmd

            screen_w, screen_h = query_window_size(
                WINDOW_NAME, (screen_w, screen_h), fullscreen=APP_FULLSCREEN
            )
            display = build_camera_module_view(
                display,
                "NGINTIL",
                last_command,
                COLOR_NGINTIL,
                screen_width=screen_w,
                screen_height=screen_h,
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
