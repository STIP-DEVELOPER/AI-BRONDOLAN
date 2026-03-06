import time

import cv2
from src.vision.brondol import BrondolDetector

from src.services.SerialService import SerialService
from src.services.DetectionLogService import DetectionLogService
from src.services.DetectionCsvService import DetectionCsvService

# ===== CONFIG =====
MODEL_PATH_BRONDOL = "src/models/brondol_ncnn_model"

SERIAL_PORT = "/dev/ttyUSB0"  # mac: /dev/tty.usbserial-xxxx
BAUDRATE = 9600

# Konfigurasi MySQL, silakan sesuaikan dengan environment Anda
MYSQL_HOST = "localhost"
MYSQL_PORT = 8889
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_DATABASE = "brondol_db"
# ==================

brondol_detector = BrondolDetector(MODEL_PATH_BRONDOL)

serial_service = None
# serial_service = SerialService(SERIAL_PORT, BAUDRATE)
# serial_service.connect()

detection_log_service = DetectionLogService(
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE,
)
detection_log_service.connect()

csv_log_service = DetectionCsvService(file_path="logs/detections.csv")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam tidak bisa dibuka")

last_command = None
last_log_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = brondol_detector.infer(frame)
        annotated = results[0].plot()

        command = brondol_detector.get_command(results[0], frame.shape)

        if command != last_command:
            print(command)
            # serial_service.send(command)
            last_command = command

        # Hitung total objek yang terdeteksi oleh YOLO pada frame ini
        boxes = results[0].boxes
        total_objects = len(boxes) if boxes is not None else 0

        # Setiap ~1 detik, simpan ke database & CSV (jika ada objek)
        now = time.time()
        if now - last_log_time >= 1.0 and total_objects > 0:
            detection_log_service.insert_detection(total_objects=total_objects)
            csv_log_service.log_detection(total_objects=total_objects)
            last_log_time = now

        cv2.putText(
            annotated,
            command if command is not None else "",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

        cv2.imshow("Brondolan", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    if serial_service is not None:
        serial_service.close()
    detection_log_service.close()
    cv2.destroyAllWindows()
