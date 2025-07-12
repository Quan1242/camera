import cv2
import mediapipe as mp

# Khởi tạo mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh về RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Nếu phát hiện tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
