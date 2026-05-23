import os
import threading
import time

import cv2
from dotenv import load_dotenv

from src.config.performance import (
    APP_FULLSCREEN,
    DISPLAY_WAIT_MS,
    INFER_EVERY_N_FRAMES,
    UI_ALTERNATE_INFER,
)
from src.services.DetectionCsvService import DetectionCsvService
from src.services.camera_manager import CameraManager, make_no_signal_frame
from src.services.serial_manager import get_shared_serial, release_shared_serial
from src.ui.cv2_dashboard import (
    build_dashboard,
    query_window_size,
    setup_opencv_window,
)
from src.vision.brondol import BrondolDetector
from src.vision.inference_runner import InferenceRunner
from src.vision.ngintil import NgintilDetector
from src.voice.agent import VoiceAgent

load_dotenv()

MODEL_PATH_NGINTIL = "src/models/ngintil_ncnn_model"
MODEL_PATH_BRONDOL = "src/models/brondol_ncnn_model"

WINDOW_NAME = "SAVIRA AI"
LOG_INTERVAL_SEC = 1.0


def main() -> None:
    print("[START] Inisialisasi kamera...", flush=True)
    camera = CameraManager(roles=["ngintil", "brondol"])
    print("[START] Memuat model YOLO...", flush=True)
    ngintil_detector = NgintilDetector(MODEL_PATH_NGINTIL)
    brondol_detector = BrondolDetector(MODEL_PATH_BRONDOL)
    csv_service = DetectionCsvService(file_path="logs/detections.csv")

    ng_runner = InferenceRunner()
    br_runner = InferenceRunner()

    serial_service = get_shared_serial()
    voice_agent = VoiceAgent(serial_service=serial_service, manage_serial=False)
    voice_thread = threading.Thread(target=voice_agent.run_loop, daemon=True)
    voice_thread.start()

    last_ngintil_command = None
    last_brondol_command = None
    last_log_time = time.time()
    brondol_object_count = 0
    frame_tick = 0

    ngintil_placeholder = make_no_signal_frame(424, 240, "NGINTIL — colokkan kamera")
    brondol_placeholder = make_no_signal_frame(424, 240, "BRONDOLAN — colokkan kamera ke-2")

    setup_opencv_window(WINDOW_NAME, fullscreen=APP_FULLSCREEN)
    screen_w, screen_h = query_window_size(WINDOW_NAME, fullscreen=APP_FULLSCREEN)

    mute_button_rect: list[tuple[int, int, int, int] | None] = [None]

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        rect = mute_button_rect[0]
        if rect is None:
            return
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            voice_agent.toggle_mute()

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Dashboard — deteksi YOLO aktif (conf dari .env / default 0.5)")
    print(f"[PERF] skip={INFER_EVERY_N_FRAMES} alternate={UI_ALTERNATE_INFER}")
    print(f"[CAMERA] {camera.get_status_text()}")

    try:
        while True:
            frame_tick += 1
            ok_n, frame_n = camera.read("ngintil")
            ok_b, frame_b = camera.read("brondol")

            cycle = frame_tick // max(1, INFER_EVERY_N_FRAMES)
            if UI_ALTERNATE_INFER:
                infer_ngintil = cycle % 2 == 0
                infer_brondol = cycle % 2 == 1
            else:
                infer_ngintil = True
                infer_brondol = True

            if ok_n and frame_n is not None and camera.is_enabled("ngintil"):

                def on_ng_cmd(cmd: str | None) -> None:
                    nonlocal last_ngintil_command
                    if cmd != last_ngintil_command and cmd is not None:
                        print(f"[NGINTIL] {cmd}")
                        if serial_service is not None:
                            serial_service.send(cmd)
                        last_ngintil_command = cmd

                ngintil_view, cmd, _ = ng_runner.run(
                    ngintil_detector,
                    frame_n,
                    get_command=ngintil_detector.get_command,
                    on_command_change=on_ng_cmd,
                    infer_this_frame=infer_ngintil or not ng_runner.has_results,
                    class_id=0,
                )
                if cmd is not None:
                    last_ngintil_command = cmd
            else:
                ngintil_view = ngintil_placeholder

            if ok_b and frame_b is not None and camera.is_enabled("brondol"):

                def on_br_cmd(cmd: str | None) -> None:
                    nonlocal last_brondol_command
                    if cmd != last_brondol_command and cmd is not None:
                        print(f"[BRONDOL] {cmd}")
                        if serial_service is not None:
                            serial_service.send(cmd)
                        last_brondol_command = cmd

                brondol_view, cmd, results = br_runner.run(
                    brondol_detector,
                    frame_b,
                    get_command=brondol_detector.get_command,
                    on_command_change=on_br_cmd,
                    infer_this_frame=infer_brondol or not br_runner.has_results,
                    class_id=None,
                )
                if cmd is not None:
                    last_brondol_command = cmd
                if results is not None and results.boxes is not None:
                    brondol_object_count = len(results.boxes)
                    now = time.time()
                    if now - last_log_time >= LOG_INTERVAL_SEC and brondol_object_count > 0:
                        csv_service.log_detection(total_objects=brondol_object_count)
                        last_log_time = now
            else:
                brondol_view = brondol_placeholder
                brondol_object_count = 0

            screen_w, screen_h = query_window_size(
                WINDOW_NAME, (screen_w, screen_h), fullscreen=APP_FULLSCREEN
            )

            voice_state = voice_agent.get_state()
            canvas, mute_button_rect[0] = build_dashboard(
                ngintil_frame=ngintil_view,
                brondol_frame=brondol_view,
                ngintil_command=last_ngintil_command,
                brondol_command=last_brondol_command,
                brondol_object_count=brondol_object_count,
                voice_state=voice_state,
                screen_width=screen_w,
                screen_height=screen_h,
            )

            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
            if key == 27:
                break
            if key in (ord("m"), ord("M")):
                voice_agent.toggle_mute()
    finally:
        voice_agent.stop()
        release_shared_serial()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
