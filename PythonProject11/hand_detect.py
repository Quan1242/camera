import cv2
import mediapipe as mp
import math

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Giới hạn 1 tay để tập trung phân tích
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def detect_grip(hand_landmarks):
    # Lấy điểm landmarks quan trọng
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]  # Cổ tay
    fingertips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],   # Ngón cái
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],  # Ngón trỏ
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],  # Ngón giữa
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],   # Ngón áp út
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]          # Ngón út
    ]
    finger_mcp = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],  # Khớp gốc ngón trỏ
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP], # Khớp gốc ngón giữa
    ]

    # Tính khoảng cách từ đầu ngón tay đến cổ tay
    tip_to_wrist_distances = [
        math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2) for tip in fingertips
    ]

    # Tính khoảng cách từ đầu ngón tay đến khớp gốc (MCP)
    tip_to_mcp_distances = [
        math.sqrt(
            (fingertips[i].x - finger_mcp[0].x)**2 +
            (fingertips[i].y - finger_mcp[0].y)**2
        ) for i in range(1, 5)  # Bỏ ngón cái (chỉ xét 4 ngón dài)
    ]

    # Ngưỡng để xác định trạng thái (có thể điều chỉnh)
    GRIP_THRESHOLD = 0.08  # Khoảng cách nhỏ => đang cầm vật
    FIST_THRESHOLD = 0.05  # Khoảng cách rất nhỏ => nắm tay chặt

    avg_tip_to_mcp = sum(tip_to_mcp_distances) / len(tip_to_mcp_distances)
    avg_tip_to_wrist = sum(tip_to_wrist_distances) / len(tip_to_wrist_distances)

    # Phân loại trạng thái
    if avg_tip_to_mcp < FIST_THRESHOLD:
        return "FIST"
    elif avg_tip_to_mcp < GRIP_THRESHOLD:
        return "GRIPPING"
    else:
        return "OPEN"

# Mở camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Chuyển đổi màu và xử lý
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ landmarks lên frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Phát hiện trạng thái cầm nắm
            grip_state = detect_grip(hand_landmarks)
            cv2.putText(
                frame, f"State: {grip_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

    # Hiển thị frame
    cv2.imshow("Grip Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()