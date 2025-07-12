import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

# Constants
VISIBILITY_THRESHOLD = 0.5
EYE_LEVEL_THRESHOLD = 0.02
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


class SimpleKalmanFilter:
    def __init__(self):
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.1
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return self.state[:2]

        # Simple predict and update
        self.state[:2] += self.state[2:]
        self.P += self.Q

        # Update
        y = measurement - self.state[:2]
        S = self.P[:2, :2] + self.R
        K = self.P[:2, :2] @ np.linalg.inv(S)

        self.state[:2] += K @ y
        self.P[:2, :2] -= K @ self.P[:2, :2]

        return self.state[:2]


class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.person_trackers = {}
        self.kalman_filters = {}
        self.next_id = 0
        self.pose_history = deque(maxlen=5)

    def _preprocess_frame(self, frame):
        """Basic preprocessing"""
        # Simple contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _detect_persons(self, frame):
        """Detect persons using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        results = pose.process(rgb_frame)
        pose.close()

        if not results.pose_landmarks:
            return []

        # Calculate bounding box
        landmarks = results.pose_landmarks.landmark
        x_coords = [lm.x * width for lm in landmarks if lm.visibility > 0.5]
        y_coords = [lm.y * height for lm in landmarks if lm.visibility > 0.5]

        if not x_coords or not y_coords:
            return []

        margin = 50
        x1 = max(0, int(min(x_coords)) - margin)
        y1 = max(0, int(min(y_coords)) - margin)
        x2 = min(width, int(max(x_coords)) + margin)
        y2 = min(height, int(max(y_coords)) + margin)

        return [(x1, y1, x2, y2)]

    def _assign_person_id(self, bbox):
        """Simple person tracking"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Find closest existing tracker
        min_dist = float('inf')
        best_id = None

        for person_id, tracker in self.person_trackers.items():
            dist = np.sqrt((center[0] - tracker['center'][0]) ** 2 +
                           (center[1] - tracker['center'][1]) ** 2)
            if dist < min_dist and dist < 150:
                min_dist = dist
                best_id = person_id

        if best_id is not None:
            self.person_trackers[best_id]['center'] = center
            self.person_trackers[best_id]['bbox'] = bbox
            return best_id
        else:
            new_id = self.next_id
            self.next_id += 1
            self.person_trackers[new_id] = {'center': center, 'bbox': bbox}
            self.kalman_filters[new_id] = {}
            return new_id

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])

        ba = a - b
        bc = c - b

        dot_product = np.dot(ba, bc)
        norms = np.linalg.norm(ba) * np.linalg.norm(bc)

        if norms < 1e-10:
            return 0.0

        cosine_angle = np.clip(dot_product / norms, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def _analyze_shooting_posture(self, points, face_results):
        """Analyze shooting posture"""
        analysis = {
            'weapon_grip': None,
            'body_stability': None,
            'gun_at_eye_level': False,
            'shooting_score': 0.0
        }

        # Check hand positions
        if all(k in points for k in ['left_wrist', 'right_wrist']):
            left_wrist = np.array([points['left_wrist'][0], points['left_wrist'][1]])
            right_wrist = np.array([points['right_wrist'][0], points['right_wrist'][1]])

            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            if wrist_distance < 0.1:
                analysis['weapon_grip'] = 'two_handed'
                analysis['shooting_score'] += 0.4
            elif wrist_distance < 0.2:
                analysis['weapon_grip'] = 'supported'
                analysis['shooting_score'] += 0.2

        # Check body stability
        if all(k in points for k in ['left_shoulder', 'right_shoulder']):
            left_shoulder = np.array([points['left_shoulder'][0], points['left_shoulder'][1]])
            right_shoulder = np.array([points['right_shoulder'][0], points['right_shoulder'][1]])

            shoulder_level = abs(left_shoulder[1] - right_shoulder[1])
            if shoulder_level < 0.05:
                analysis['body_stability'] = 'stable'
                analysis['shooting_score'] += 0.3

        # Check eye level aiming
        if face_results and face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            eye_level = (left_eye.y + right_eye.y) / 2

            if 'left_wrist' in points and 'right_wrist' in points:
                avg_wrist_y = (points['left_wrist'][1] + points['right_wrist'][1]) / 2
                if abs(avg_wrist_y - eye_level) < EYE_LEVEL_THRESHOLD:
                    analysis['gun_at_eye_level'] = True
                    analysis['shooting_score'] += 0.3

        return analysis

    def _analyze_person_pose(self, person_frame, bbox, face_results, person_id):
        """Analyze single person pose"""
        rgb_person = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)

        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        results = pose.process(rgb_person)
        pose.close()

        if not results.pose_landmarks:
            return None

        # Extract key points
        landmarks = results.pose_landmarks.landmark
        x1, y1, x2, y2 = bbox

        key_points = {
            'nose': 0, 'left_eye': 2, 'right_eye': 5,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24
        }

        points = {}
        for name, idx in key_points.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > VISIBILITY_THRESHOLD:
                    # Convert to global coordinates
                    global_x = (lm.x * (x2 - x1) + x1) / 1280.0
                    global_y = (lm.y * (y2 - y1) + y1) / 720.0
                    points[name] = (global_x, global_y, lm.visibility)

        if len(points) < 5:
            return None

        # Calculate angles
        angles = {}
        if all(k in points for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles['left_elbow'] = self._calculate_angle(
                points['left_shoulder'], points['left_elbow'], points['left_wrist'])

        if all(k in points for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles['right_elbow'] = self._calculate_angle(
                points['right_shoulder'], points['right_elbow'], points['right_wrist'])

        # Analyze shooting posture
        shooting_analysis = self._analyze_shooting_posture(points, face_results)

        return {
            'person_id': person_id,
            'bbox': bbox,
            'landmarks': points,
            'angles': angles,
            'shooting_analysis': shooting_analysis
        }

    def analyze_frame(self, frame):
        """Main analysis function"""
        start_time = time.time()

        # Preprocess
        processed_frame = self._preprocess_frame(frame)
        height, width = frame.shape[:2]

        # Detect persons
        person_bboxes = self._detect_persons(processed_frame)

        # Process faces
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_frame)

        # Analyze each person
        results = {
            'persons': {},
            'frame_info': {
                'width': width,
                'height': height,
                'person_count': len(person_bboxes),
                'processing_time': time.time() - start_time
            }
        }

        for bbox in person_bboxes:
            person_id = self._assign_person_id(bbox)

            x1, y1, x2, y2 = bbox
            person_frame = processed_frame[y1:y2, x1:x2]

            if person_frame.size == 0:
                continue

            person_results = self._analyze_person_pose(person_frame, bbox, face_results, person_id)

            if person_results:
                results['persons'][person_id] = person_results

        return results

    def visualize_results(self, frame, results):
        """Visualize results"""
        vis_frame = frame.copy()

        for person_id, person_data in results['persons'].items():
            color = COLORS[person_id % len(COLORS)]

            # Draw bounding box
            bbox = person_data['bbox']
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw person ID
            cv2.putText(vis_frame, f"Person {person_id}",
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw landmarks
            landmarks = person_data['landmarks']
            for point_name, (x, y, visibility) in landmarks.items():
                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])
                cv2.circle(vis_frame, (px, py), 3, color, -1)

            # Draw shooting analysis
            shooting = person_data['shooting_analysis']
            y_offset = bbox[3] + 20

            if shooting['weapon_grip']:
                cv2.putText(vis_frame, f"Grip: {shooting['weapon_grip']}",
                            (bbox[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 15

            cv2.putText(vis_frame, f"Score: {shooting['shooting_score']:.2f}",
                        (bbox[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Frame info
        info = results['frame_info']
        cv2.putText(vis_frame, f"Persons: {info['person_count']} | Time: {info['processing_time']:.3f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis_frame

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        self.person_trackers.clear()
        self.kalman_filters.clear()


# Usage example
def main():
    analyzer = PoseAnalyzer()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = analyzer.analyze_frame(frame)
        vis_frame = analyzer.visualize_results(frame, results)

        cv2.imshow('Pose Analysis', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    analyzer.cleanup()


if __name__ == "__main__":
    main()