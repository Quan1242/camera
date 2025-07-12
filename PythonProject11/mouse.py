import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy

class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList, yList = [], []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20),
                              (0, 255, 0), 2)
        return self.lmList, bbox

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector = handDetector(detectionCon=0.6, trackCon=0.6)
    screenW, screenH = autopy.screen.size()
    wCam, hCam = 640, 480
    frameR = 100  # vùng đệm tránh tràn màn hình
    smoothening = 5
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    lastClickTime = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1], lmList[8][2]  # ngón trỏ
            x2, y2 = lmList[4][1], lmList[4][2]  # ngón cái

            # Convert tọa độ sang màn hình
            screenX = np.interp(x1, (frameR, wCam-frameR), (0, screenW))
            screenY = np.interp(y1, (frameR, hCam-frameR), (0, screenH))

            # Smooth di chuyển để không giật
            clocX = plocX + (screenX - plocX) / smoothening
            clocY = plocY + (screenY - plocY) / smoothening

            # Move chuột
            autopy.mouse.move(clocX, clocY)
            plocX, plocY = clocX, clocY

            # Tính khoảng cách giữa ngón cái và trỏ để click
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 30 and time.time() - lastClickTime > 0.3:
                autopy.mouse.click()
                lastClickTime = time.time()

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Hand Mouse Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
