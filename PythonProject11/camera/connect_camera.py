import cv2

class CameraConnector:
    def __init__(self, source):
        self.source = source
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Không thể kết nối tới nguồn {self.source}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
