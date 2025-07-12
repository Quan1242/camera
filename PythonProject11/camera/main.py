import customtkinter as ctk
import requests
import numpy as np
import cv2
from threading import Thread
from connect_camera import CameraConnector
from take_infomation import MultiPersonPoseAnalyzer

def start_camera(source_type, ip, username, password):
    analyzer = MultiPersonPoseAnalyzer()

    if source_type == "laptop":
        cap = cv2.VideoCapture(0)

    elif source_type == "axis_http":
        url = f"http://{username}:{password}@{ip}/axis-cgi/jpg/image.cgi"

    elif source_type == "axis_rtsp":
        rtsp_url = f"rtsp://{username}:{password}@{ip}/axis-media/media.amp"
        connector = CameraConnector(rtsp_url)
        connector.connect()

    while True:
        if source_type == "laptop":
            ret, frame = cap.read()
            if not ret:
                break

        elif source_type == "axis_http":
            try:
                resp = requests.get(url, stream=True, timeout=3)
                if resp.status_code == 200:
                    img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                else:
                    break
            except:
                break

        elif source_type == "axis_rtsp":
            frame = connector.get_frame()
            if frame is None:
                break

        processed_frame, results = analyzer.analyze(frame)
        cv2.imshow("Camera Viewer", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if source_type == "laptop":
        cap.release()
    elif source_type == "axis_rtsp":
        connector.release()

    analyzer.release()
    cv2.destroyAllWindows()

def connect_camera():
    source_type = camera_source.get()
    ip = ip_entry.get().strip()
    username = username_entry.get().strip()
    password = password_entry.get().strip()

    if source_type != "laptop" and (not ip or not username or not password):
        ctk.CTkMessageBox(title="Lỗi", message="Nhập đầy đủ thông tin")
        return

    thread = Thread(target=start_camera, args=(source_type, ip, username, password))
    thread.daemon = True
    thread.start()

def main():
    global camera_source, ip_entry, username_entry, password_entry

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("Chọn nguồn Camera")
    app.geometry("400x350")

    camera_source = ctk.StringVar(value="laptop")

    ctk.CTkLabel(app, text="Chọn nguồn camera:").pack(pady=5)
    ctk.CTkRadioButton(app, text="Laptop Camera", variable=camera_source, value="laptop").pack(pady=2)
    ctk.CTkRadioButton(app, text="AXIS HTTP snapshot", variable=camera_source, value="axis_http").pack(pady=2)
    ctk.CTkRadioButton(app, text="AXIS RTSP stream", variable=camera_source, value="axis_rtsp").pack(pady=2)

    ip_entry = ctk.CTkEntry(app, placeholder_text="Địa chỉ IP")
    ip_entry.pack(pady=10)

    username_entry = ctk.CTkEntry(app, placeholder_text="Username")
    username_entry.pack(pady=10)

    password_entry = ctk.CTkEntry(app, placeholder_text="Password", show="*")
    password_entry.pack(pady=10)

    connect_button = ctk.CTkButton(app, text="Kết nối Camera", command=connect_camera)
    connect_button.pack(pady=20)

    app.mainloop()

if __name__ == "__main__":
    main()