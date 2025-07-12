import cv2
import mediapipe as mp
import math
import numpy as np
import time
import threading
from queue import Queue

# Thử import PiCamera (cho Raspberry Pi)
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray

    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("PiCamera not available, using OpenCV VideoCapture")

# Thử import RPi.GPIO cho LED indicator (optional)
try:
    import RPi.GPIO as GPIO

    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class RaspberryPiPoseChecker:
    def __init__(self, max_eye_wrist_ratio=0.35, min_arm_length=0.13, use_pi_camera=True):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.max_eye_wrist_ratio = max_eye_wrist_ratio
        self.min_arm_length = min_arm_length
        self.use_pi_camera = use_pi_camera and PI_CAMERA_AVAILABLE

        # Tối ưu cho Raspberry Pi - giảm độ phức tạp
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Giảm xuống 0 cho RPi
            enable_segmentation=False,
            min_detection_confidence=0.6,  # Giảm threshold
            min_tracking_confidence=0.4
        )

        # Tối ưu performance cho RPi
        self._last_result = None
        self._frame_count = 0
        self._process_every_n_frames = 3  # Xử lý mỗi 3 frame cho RPi

        # Threading để tối ưu camera
        self.frame_queue = Queue(maxsize=2)
        self.camera_thread = None
        self.running = False

        # GPIO setup cho LED indicator (optional)
        self.led_pin = 18
        if GPIO_AVAILABLE:
            self.setup_gpio()

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def setup_gpio(self):
        """Setup GPIO cho LED indicator"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.output(self.led_pin, GPIO.LOW)
            print("GPIO setup complete")
        except Exception as e:
            print(f"GPIO setup failed: {e}")

    def control_led(self, state):
        """Điều khiển LED indicator"""
        if GPIO_AVAILABLE:
            try:
                GPIO.output(self.led_pin, GPIO.HIGH if state else GPIO.LOW)
            except:
                pass

    def distance(self, point1, point2):
        """Tối ưu: Sử dụng numpy cho tính toán nhanh hơn"""
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def check_pose(self, landmarks):
        """Kiểm tra tư thế bắn súng"""
        try:
            lm = landmarks
            right_eye = lm[self.mp_pose.PoseLandmark.RIGHT_EYE]
            right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Kiểm tra visibility
            if (right_eye.visibility < 0.4 or  # Giảm threshold cho RPi
                    right_wrist.visibility < 0.4 or
                    right_shoulder.visibility < 0.4):
                return "Unable to detect pose clearly"

            arm_length = self.distance(right_shoulder, right_wrist)

            if arm_length < self.min_arm_length:
                return "Arm too close or not detected properly"

            ratio_eye_wrist = self.distance(right_eye, right_wrist) / arm_length

            is_right_pose = ratio_eye_wrist < self.max_eye_wrist_ratio

            # Điều khiển LED
            self.control_led(is_right_pose)

            return "Right pose detected" if is_right_pose else "Wrong pose detected"

        except (IndexError, AttributeError) as e:
            return f"Error detecting pose: {str(e)}"

    def camera_thread_func(self):
        """Thread xử lý camera riêng biệt"""
        if self.use_pi_camera:
            self.pi_camera_capture()
        else:
            self.opencv_camera_capture()

    def pi_camera_capture(self):
        """Sử dụng PiCamera để capture"""
        camera = PiCamera()
        camera.resolution = (640, 480)  # Resolution thấp cho RPi
        camera.framerate = 24  # FPS thấp hơn
        camera.rotation = 0  # Xoay camera nếu cần

        # Điều chỉnh camera settings
        camera.brightness = 50
        camera.contrast = 10
        camera.exposure_mode = 'auto'

        rawCapture = PiRGBArray(camera, size=(640, 480))

        print("PiCamera initialized")
        time.sleep(2)  # Warm up camera

        try:
            for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                if not self.running:
                    break

                image = frame.array

                # Đưa frame vào queue
                if not self.frame_queue.full():
                    self.frame_queue.put(image)

                rawCapture.truncate(0)

        except Exception as e:
            print(f"PiCamera error: {e}")
        finally:
            camera.close()

    def opencv_camera_capture(self):
        """Sử dụng OpenCV để capture"""
        cap = cv2.VideoCapture(0)

        # Tối ưu settings cho RPi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("OpenCV camera initialized")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Đưa frame vào queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

        except Exception as e:
            print(f"OpenCV camera error: {e}")
        finally:
            cap.release()

    def process_frame(self, frame):
        """Xử lý frame với tối ưu cho RPi"""
        # Chuyển đổi BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize nhỏ hơn để tăng tốc độ xử lý
        height, width = rgb_frame.shape[:2]
        if width > 320:
            scale = 320 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))

        # Chỉ xử lý mỗi n frames
        if self._frame_count % self._process_every_n_frames == 0:
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks:
                self._last_result = {
                    'landmarks': results.pose_landmarks,
                    'pose_result': self.check_pose(results.pose_landmarks.landmark)
                }

        self._frame_count += 1
        return self._last_result

    def calculate_fps(self):
        """Tính FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Mỗi 30 frame
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time

    def draw_results(self, frame, result_data):
        """Vẽ kết quả với tối ưu cho RPi"""
        if not result_data:
            return frame

        # Vẽ skeleton với độ dày nhỏ hơn
        self.mp_drawing.draw_landmarks(
            frame,
            result_data['landmarks'],
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
        )

        # Vẽ text kết quả
        pose_result = result_data['pose_result']
        color = (0, 255, 0) if "Right pose detected" in pose_result else (0, 0, 255)

        # Text nhỏ hơn cho RPi
        cv2.putText(frame, pose_result, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Hiển thị FPS và frame count
        cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Frame: {self._frame_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Hiển thị camera type
        camera_type = "PiCamera" if self.use_pi_camera else "OpenCV"
        cv2.putText(frame, f"Camera: {camera_type}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def run(self):
        """Chạy chương trình chính"""
        print("Starting Raspberry Pi Pose Checker...")
        print(f"Using {'PiCamera' if self.use_pi_camera else 'OpenCV'} for capture")
        print("Press ESC or 'q' to exit")

        self.running = True
        self.fps_start_time = time.time()

        # Khởi động camera thread
        self.camera_thread = threading.Thread(target=self.camera_thread_func)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        try:
            while self.running:
                # Lấy frame từ queue
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()

                    # Xử lý frame
                    result_data = self.process_frame(frame)

                    # Vẽ kết quả
                    frame = self.draw_results(frame, result_data)

                    # Tính FPS
                    self.calculate_fps()

                    # Hiển thị
                    cv2.imshow("RPi Pose Checker", frame)

                # Kiểm tra phím thoát
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        self.running = False

        # Đợi camera thread kết thúc
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)

        # Cleanup
        cv2.destroyAllWindows()
        self.pose.close()

        # Cleanup GPIO
        if GPIO_AVAILABLE:
            GPIO.cleanup()

        print("Pose checker stopped")


# Chạy chương trình
if __name__ == "__main__":
    # Tự động detect PiCamera hoặc dùng OpenCV
    checker = RaspberryPiPoseChecker(use_pi_camera=True)
    checker.run()