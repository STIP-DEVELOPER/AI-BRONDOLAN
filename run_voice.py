import threading

import cv2

from src.config.performance import APP_FULLSCREEN, DISPLAY_WAIT_MS
from src.ui.cv2_dashboard import (
    build_voice_dashboard,
    query_window_size,
    setup_opencv_window,
)
from src.voice.agent import VoiceAgent

# Untuk UI terpadu (kamera + voice), jalankan: python run_ui.py

WINDOW_NAME = "SAVIRA AI — Voice"


def main() -> None:
    agent = VoiceAgent()
    print("SAVIRA AI — Voice")
    print("Tanya apa saja, atau ucapkan: MAJU / MUNDUR / KIRI / KANAN / STOP / AMBIL / CHECK")
    if agent.serial_connected:
        port = agent._serial.port if agent._serial else "?"
        print(f"Serial Arduino: terhubung ({port})")
    else:
        print("Serial Arduino: tidak terdeteksi — colokkan USB Arduino")
    print("ESC keluar  |  M mute  |  Klik tombol MUTE di layar\n")

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
            agent.toggle_mute()

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    voice_thread = threading.Thread(target=agent.run_loop, daemon=True)
    voice_thread.start()

    try:
        while True:
            screen_w, screen_h = query_window_size(
                WINDOW_NAME, (screen_w, screen_h), fullscreen=APP_FULLSCREEN
            )
            canvas, mute_button_rect[0] = build_voice_dashboard(
                agent.get_state(),
                screen_width=screen_w,
                screen_height=screen_h,
            )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
            if key == 27:
                break
            if key in (ord("m"), ord("M")):
                agent.toggle_mute()
    except KeyboardInterrupt:
        print("\nKeluar")
    finally:
        agent.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
