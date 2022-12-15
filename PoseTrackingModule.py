import cv2
import mediapipe as mp
import time
import math
from typing import Tuple, Union

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
    
    def _normalized_to_pixel_coordinates(
        self,normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
      """Converts normalized value pair to pixel coordinates."""

      # Checks if the float value is between 0 and 1.
      def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

      if not (is_valid_normalized_value(normalized_x) and
              is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px,y_px, False
      x_px = min(math.floor(normalized_x * image_width), image_width - 1)
      y_px = min(math.floor(normalized_y * image_height), image_height - 1)
      return x_px, y_px , True
    
    def getPixelLandmarks(self,landmarks,color_img):
        image_rows, image_cols, _ = color_img.shape
        pixLandmarks=[]
        if landmarks:
            for landmark in landmarks:
                if (landmark[0] and landmark[1]):
                    x_px,y_px,isLandmarkVisible = self._normalized_to_pixel_coordinates(landmark[0],landmark[1],image_cols, image_rows)
                pixLandmarks.append([x_px,y_px,isLandmarkVisible])
        return pixLandmarks
    
    def findPoseAndDrawLandmarksWithYolo(self, color_img, depth_img,x1,y1,xw,yh,):
        cropped_color_img, cropped_depth_img=self.findPoseColorAndDepth(color_img[y1:yh,x1:xw], depth_img[y1:yh,x1:xw])
        landmarks=[]
        if self.results.pose_landmarks:
            for data_point in self.results.pose_landmarks.landmark:
                landmarks.append([data_point.x,data_point.y,data_point.visibility])
        pixLandmarks=self.getPixelLandmarks(landmarks,color_img[y1:yh,x1:xw])
        for pixLandmark in pixLandmarks:
            pixLandmark[0]=pixLandmark[0]+x1
            pixLandmark[1]=pixLandmark[1]+y1
        return color_img, depth_img, pixLandmarks

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
