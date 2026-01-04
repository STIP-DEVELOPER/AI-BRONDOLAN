import cv2
from src.services.SerialService import SerialService
from src.vision.brondol import BrondolDetector  

# ===== CONFIG =====
MODEL_PATH = "src/models/brondol_ncnn_model"
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 9600
# ==================

detector = BrondolDetector(MODEL_PATH)

# serial_service = SerialService(SERIAL_PORT, BAUDRATE)
# serial_service.connect()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam tidak bisa dibuka")

last_take_sent = False  # untuk mencegah spam TAKE

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.infer(frame)
    annotated = results[0].plot()

    command = detector.get_command(results[0], frame.shape)

    # if command == "TAKE" and not last_take_sent:
    #     print("TAKE")
    #     serial_service.send("TAKE")
    #     last_take_sent = True
    # elif command is None:
    #     last_take_sent = False  

    if command:
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
