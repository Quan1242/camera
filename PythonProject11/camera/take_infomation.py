import cv2
import numpy as np
import mediapipe as mp
import math
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

# Constants để tránh magic numbers
VISIBILITY_THRESHOLD = 0.5
EYE_LEVEL_THRESHOLD = 0.03
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 2

# Face landmarks constants
NOSE_TIP_IDX = 1
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# YOLO constants
PERSON_CLASS_ID = 0
YOLO_CONFIDENCE_THRESHOLD = 0.5
CROP_PADDING = 20  # Padding cho crop region


class MultiPersonPoseAnalyzer:
    def __init__(self, max_people=4, yolo_model_path="yolov8n.pt"):
        """Khởi tạo bộ phân tích tư thế cho nhiều người với YOLO optimization"""
        self.max_people = max_people
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)

        # Cache các pose landmark indices để tránh lookup
        self._pose_indices = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP.value,
        }

        # Pre-allocate arrays để tối ưu memory
        self._temp_vectors = {
            'ba': np.zeros(2),
            'bc': np.zeros(2),
        }

        # Pre-compute colors
        self._colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
        }

        # Sử dụng Pose với static_image_mode=False để xử lý nhanh hơn trên crop regions
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Thay đổi để xử lý nhanh hơn
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Giảm xuống 1 vì chỉ xử lý 1 người mỗi crop
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def _detect_persons_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Phát hiện người trong khung hình bằng YOLO"""
        results = self.yolo_model(frame, verbose=False)
        person_boxes = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Chỉ lấy class person (class_id = 0)
                    if box.cls == PERSON_CLASS_ID and box.conf > YOLO_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        person_boxes.append((x1, y1, x2, y2))

        return person_boxes[:self.max_people]  # Giới hạn số người

    def _expand_bbox(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int],
                     padding: int = CROP_PADDING) -> Tuple[int, int, int, int]:
        """Mở rộng bounding box với padding"""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return (x1, y1, x2, y2)

    def _crop_person_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[
        np.ndarray, Tuple[int, int]]:
        """Crop vùng người từ khung hình"""
        x1, y1, x2, y2 = self._expand_bbox(bbox, frame.shape)
        cropped = frame[y1:y2, x1:x2]
        return cropped, (x1, y1)

    def _transform_landmarks_to_original(self, landmarks, crop_offset: Tuple[int, int],
                                         crop_size: Tuple[int, int]) -> None:
        """Chuyển đổi landmarks từ crop coordinate về original coordinate"""
        offset_x, offset_y = crop_offset
        crop_h, crop_w = crop_size

        for landmark in landmarks.landmark:
            # Chuyển từ normalized coordinate trong crop về pixel coordinate
            pixel_x = landmark.x * crop_w + offset_x
            pixel_y = landmark.y * crop_h + offset_y

            # Chuyển về normalized coordinate trong original frame
            landmark.x = pixel_x / self.original_width
            landmark.y = pixel_y / self.original_height

    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Tính góc giữa 3 điểm - tối ưu với pre-allocated arrays"""
        # Sử dụng pre-allocated arrays thay vì tạo mới
        self._temp_vectors['ba'][0] = a[0] - b[0]
        self._temp_vectors['ba'][1] = a[1] - b[1]
        self._temp_vectors['bc'][0] = c[0] - b[0]
        self._temp_vectors['bc'][1] = c[1] - b[1]

        # Tính toán nhanh hơn
        dot_product = np.dot(self._temp_vectors['ba'], self._temp_vectors['bc'])
        norm_ba = np.linalg.norm(self._temp_vectors['ba'])
        norm_bc = np.linalg.norm(self._temp_vectors['bc'])

        cosine_angle = dot_product / (norm_ba * norm_bc)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))
        return angle

    def analyze(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Phân tích tư thế nhiều người trong khung hình - tối ưu với YOLO"""
        if frame is None:
            return None, []

        # Lưu kích thước gốc để transform landmarks
        self.original_height, self.original_width = frame.shape[:2]
        annotated_frame = frame.copy()
        all_results = []

        # Bước 1: Phát hiện người bằng YOLO
        person_boxes = self._detect_persons_yolo(frame)

        if not person_boxes:
            return annotated_frame, []

        # Bước 2: Xử lý từng người riêng biệt
        for i, bbox in enumerate(person_boxes):
            # Crop vùng người
            cropped_frame, crop_offset = self._crop_person_region(frame, bbox)

            if cropped_frame.size == 0:
                continue

            # Chuyển đổi màu cho crop
            cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            # Xử lý pose và face trên crop
            pose_results = self.pose.process(cropped_rgb)
            face_results = self.face_mesh.process(cropped_rgb)

            if not pose_results.pose_landmarks:
                continue

            # Transform landmarks về coordinate gốc
            self._transform_landmarks_to_original(
                pose_results.pose_landmarks,
                crop_offset,
                cropped_frame.shape[:2]
            )

            # Xử lý face landmarks nếu có
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    self._transform_landmarks_to_original(
                        face_landmarks,
                        crop_offset,
                        cropped_frame.shape[:2]
                    )

            # Phân tích tư thế (giữ nguyên logic cũ)
            result = self._analyze_single_person(pose_results, face_results, i)
            all_results.append(result)

            # Vẽ landmarks trên frame gốc
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            # Vẽ bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self._colors['blue'], 2)
            cv2.putText(annotated_frame, f"Person {i + 1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._colors['blue'], 2)

        # Vẽ kết quả phân tích cho tất cả người
        self._draw_all_analysis_results(annotated_frame, all_results)

        return annotated_frame, all_results

    def _analyze_single_person(self, pose_results, face_results, person_id: int) -> Dict:
        """Phân tích tư thế cho một người - giữ nguyên logic cũ"""
        landmarks = pose_results.pose_landmarks.landmark

        # Pre-compute face landmarks dict
        face_landmarks_dict = {}
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                nose_tip = face_landmarks.landmark[NOSE_TIP_IDX]
                face_landmarks_dict[(nose_tip.x, nose_tip.y)] = face_landmarks

        # Optimized get_xy function
        def get_xy_fast(landmark_key):
            idx = self._pose_indices[landmark_key]
            lm = landmarks[idx]
            return (lm.x, lm.y) if lm.visibility > VISIBILITY_THRESHOLD else None

        # Lấy tất cả điểm cùng lúc
        points = {
            'ls': get_xy_fast('left_shoulder'),
            'rs': get_xy_fast('right_shoulder'),
            'le': get_xy_fast('left_elbow'),
            're': get_xy_fast('right_elbow'),
            'lw': get_xy_fast('left_wrist'),
            'rw': get_xy_fast('right_wrist'),
            'lh': get_xy_fast('left_hip'),
        }

        # Pre-allocate result dict
        result = {
            'person_id': person_id,
            'left_elbow_angle': None,
            'right_elbow_angle': None,
            'body_lean_angle': None,
            'shoulder_diff': None,
            'wrist_diff': None,
            'wrist_eye_diff': None,
            'gun_at_eye_level': False
        }

        # Check if all points are valid
        if all(point is not None for point in points.values()):
            # Batch calculations
            result['left_elbow_angle'] = self.calculate_angle(points['ls'], points['le'], points['lw'])
            result['right_elbow_angle'] = self.calculate_angle(points['rs'], points['re'], points['rw'])
            result['body_lean_angle'] = math.degrees(math.atan2(points['lh'][1] - points['ls'][1],
                                                                points['lh'][0] - points['ls'][0]))
            result['shoulder_diff'] = abs(points['ls'][1] - points['rs'][1])
            result['wrist_diff'] = abs(points['lw'][1] - points['rw'][1])

            # Face processing
            if face_landmarks_dict:
                shoulder_center = ((points['ls'][0] + points['rs'][0]) * 0.5,
                                   (points['ls'][1] + points['rs'][1]) * 0.5)

                closest_face = min(face_landmarks_dict.items(),
                                   key=lambda item: math.dist(shoulder_center, item[0]))
                face_lms = closest_face[1].landmark

                left_eye = (face_lms[LEFT_EYE_IDX].x, face_lms[LEFT_EYE_IDX].y)
                right_eye = (face_lms[RIGHT_EYE_IDX].x, face_lms[RIGHT_EYE_IDX].y)
                eye_center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)

                wrist_eye_diff = (points['lw'][1] + points['rw'][1]) * 0.5 - eye_center[1]
                result['wrist_eye_diff'] = wrist_eye_diff
                result['gun_at_eye_level'] = abs(wrist_eye_diff) < EYE_LEVEL_THRESHOLD

        return result

    def _draw_analysis_results(self, frame: np.ndarray, result: Dict):
        """Vẽ kết quả phân tích lên frame - tối ưu"""
        y_offset = 30
        line_height = 30

        for key, val in result.items():
            if val is not None and key not in ['gun_at_eye_level', 'person_id']:
                # Pre-format text
                title = key.replace('_', ' ').title()
                unit = ' deg' if 'angle' in key else ''
                text = f"{title}: {val:.1f}{unit}"

                color = self._colors['green'] if not (key == 'wrist_eye_diff' and abs(val) < EYE_LEVEL_THRESHOLD) else \
                    self._colors['red']
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += line_height

            if key == 'gun_at_eye_level' and val:
                cv2.putText(frame, f"PERSON {result['person_id'] + 1}: GUN AT EYE LEVEL!",
                            (frame.shape[1] // 2 - 200, 50 + result['person_id'] * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, self._colors['red'], 2)

    def _draw_all_analysis_results(self, frame: np.ndarray, all_results: List[Dict]):
        """Vẽ kết quả phân tích cho tất cả người"""
        for i, result in enumerate(all_results):
            # Vẽ thông tin cơ bản ở góc trái
            y_start = 30 + i * 200  # Mỗi người cách nhau 200px

            # Header cho mỗi người
            cv2.putText(frame, f"PERSON {i + 1}:", (10, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self._colors['yellow'], 2)

            y_offset = y_start + 30
            line_height = 25

            for key, val in result.items():
                if val is not None and key not in ['gun_at_eye_level', 'person_id']:
                    title = key.replace('_', ' ').title()
                    unit = ' deg' if 'angle' in key else ''
                    text = f"{title}: {val:.1f}{unit}"

                    color = self._colors['green'] if not (
                                key == 'wrist_eye_diff' and abs(val) < EYE_LEVEL_THRESHOLD) else \
                        self._colors['red']
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += line_height

                if key == 'gun_at_eye_level' and val:
                    cv2.putText(frame, f"PERSON {i + 1}: GUN AT EYE LEVEL!",
                                (frame.shape[1] // 2 - 200, 50 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, self._colors['red'], 2)

    def release(self):
        """Giải phóng tài nguyên"""
        self.pose.close()
        self.face_mesh.close()

    def visualize_results(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Vẽ kết quả pose detection lên frame"""
        if not results:
            return frame

        # Tạo bản sao của frame gốc
        output_frame = frame.copy()

        # Vẽ bounding box và keypoints cho mỗi người
        for person_id, data in results.items():
            # Vẽ bounding box
            x1, y1, x2, y2 = data['bbox']
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ keypoints nếu có
            if 'points' in data:
                for name, (x, y, vis) in data['points'].items():
                    if vis > 0.5:  # Chỉ vẽ điểm có độ tin cậy > 50%
                        cv2.circle(output_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        return output_frame
