import cv2
import numpy as np


class CameraDisplay:
    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def start_camera(self):
        """Start camera connection"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Cannot open camera source: {self.source}")
            return False
        return True

    def add_center_crosshair(self, frame):
        """Add center crosshair to frame"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Draw crosshair lines
        line_length = 25
        color = (0, 0, 255)  # Red

   

        # Center dot
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

        return frame

    def display_camera(self):
        """Display camera feed with center crosshair"""
        if not self.start_camera():
            return

        print("\nCamera Controls:")
        print("ESC or Q: Quit")
        print("C: Toggle center crosshair")

        show_center = True

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot read frame from camera")
                break

            frame = cv2.flip(frame, 1)  # Mirror effect

            if show_center:
                frame = self.add_center_crosshair(frame)

            cv2.imshow('Camera View - ESC to quit', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                break
            elif key == ord('c'):  # Toggle center
                show_center = not show_center
                status = "ON" if show_center else "OFF"
                print(f"Center crosshair: {status}")

        self.cap.release()
        cv2.destroyAllWindows()


def list_cameras():
    """List available USB cameras"""
    print("Checking for USB cameras...")
    available = []
    for i in range(5):  # Check first 5 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    if available:
        print(f"Found {len(available)} cameras: {available}")
    else:
        print("No USB cameras found")
    return available


def main():
    print("=== SIMPLE CAMERA VIEWER ===")
    cameras = list_cameras()
    if not cameras:
        return

    # Use first available camera
    camera = CameraDisplay(cameras[0])
    camera.display_camera()


if __name__ == "__main__":
    main()