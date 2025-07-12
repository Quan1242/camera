import gradio as gr
import cv2
import numpy as np
from camera.test import PoseAnalyzer


class PoseApp:
    def __init__(self, yolo_model_path=None):
        self.analyzer = PoseAnalyzer()

    def process_frame(self, frame):
        """Xử lý từng frame webcam"""
        if frame is None:
            return None

        # Chuyển đổi từ RGB (Gradio) sang BGR (OpenCV)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Phân tích pose
        results = self.analyzer.analyze_frame(frame_bgr)

        # Vẽ kết quả lên frame (nếu có phương thức visualize_results)
        if hasattr(self.analyzer, 'visualize_results'):
            output_frame = self.analyzer.visualize_results(frame_bgr, results)
        else:
            # Fallback nếu không có phương thức visualize
            output_frame = frame_bgr
            for person_id, data in results.items():
                x1, y1, x2, y2 = data['bbox']
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Chuyển lại sang RGB để hiển thị trên Gradio
        return cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)


# Tạo giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Real-time Pose Detection")

    with gr.Row():
        webcam_input = gr.Image(source="webcam", streaming=True, label="Webcam Live")
        webcam_output = gr.Image(label="Pose Detection Result")

    webcam_input.stream(
        fn=PoseApp().process_frame,
        inputs=webcam_input,
        outputs=webcam_output
    )

if __name__ == "__main__":
    demo.launch(
        server_port=8080,
        enable_queue=True
    )