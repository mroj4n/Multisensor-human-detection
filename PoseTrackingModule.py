import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, maxpose=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxpose = maxpose
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findpose(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            for poseLmS in self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, poseLmS,
                                               self.mp_pose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, humanNo=0, draw=False):

        lmList = []
        if self.results.pose_landmarks:
            Human = self.results.pose_landmarks[humanNo]
            for id, lm in enumerate(Human.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findpose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()