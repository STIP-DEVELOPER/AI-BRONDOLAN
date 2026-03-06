import mysql.connector
from mysql.connector import Error


class DetectionLogService:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "brondol_db",
    ):
        self._config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }
        self._connection = None
        self._cursor = None

    def connect(self) -> None:
        if self._connection is not None and self._connection.is_connected():
            return

        try:
            self._connection = mysql.connector.connect(**self._config)
            self._cursor = self._connection.cursor()
            self._cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_objects INT NOT NULL
                )
                """
            )
            self._connection.commit()
        except Error as exc:
            raise RuntimeError(f"Gagal konek ke MySQL: {exc}") from exc

    def insert_detection(self, total_objects: int) -> None:
        if total_objects is None:
            return

        if self._connection is None or not self._connection.is_connected():
            self.connect()

        query = "INSERT INTO detections (total_objects) VALUES (%s)"
        try:
            self._cursor.execute(query, (total_objects,))
            self._connection.commit()
        except Error as exc:
            print(f"[MYSQL] Gagal insert detection: {exc}")

    def close(self) -> None:
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None

        if self._connection is not None and self._connection.is_connected():
            self._connection.close()
            self._connection = None
