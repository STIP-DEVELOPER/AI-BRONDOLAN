import cv2
from src.vision.ngintil import NgintilDetector
# from src.vision.brondol import BrondolDetector 

from src.services.SerialService import SerialService

# ===== CONFIG =====
MODEL_PATH_NGINTIL = "src/models/ngintil_ncnn_model"
# MODEL_PATH_BRONDOL = "src/models/brondol_ncnn_model"

SERIAL_PORT = "/dev/ttyUSB0"  # mac: /dev/tty.usbserial-xxxx
BAUDRATE = 9600
# ==================

ngintil_detector = NgintilDetector(MODEL_PATH_NGINTIL)
# brondol_detector = BrondolDetector(MODEL_PATH_BRONDOL)

# serial_service = SerialService(SERIAL_PORT, BAUDRATE)
# serial_service.connect()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam tidak bisa dibuka")

last_command = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = ngintil_detector.infer(frame)
    annotated = results[0].plot()

    command = ngintil_detector.get_command(results[0], frame.shape)

    # kirim hanya jika berubah (anti spam)
    if command != last_command:
        print(command)
        # serial_service.send(command)
        last_command = command

    cv2.putText(
        annotated,
        command,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
    )

    cv2.imshow("Brondolan", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
serial_service.close()
cv2.destroyAllWindows()
