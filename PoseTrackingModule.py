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
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findpose(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        imgRGB.flags.writeable = True
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=False):

        lmList = []
        if self.results.pose_landmarks:
            Human = self.results.pose_landmarks
            for id, lm in enumerate(Human.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList

    def findPoseColorAndDepth(self, color_img, depth_img, draw=False):
        color_img = self.findpose(color_img, True)
        depth_img.flags.writeable = True
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(depth_img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return color_img, depth_img

    def findPoseAndDrawLandmarks(self, color_img, depth_img):
        color_img, depth_img=self.findPoseColorAndDepth(color_img, depth_img)
        landmarks=[]
        if self.results.pose_landmarks:
            for data_point in self.results.pose_landmarks.landmark:
                landmarks.append([data_point.x,data_point.y,data_point.visibility])
        return color_img, depth_img, landmarks


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
