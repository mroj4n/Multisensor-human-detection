import numpy as np
import cv2

class DepthDetector():
    def __init__(self):
        pass
    def detect(self,landmarks,depth_img):
        for landmark in landmarks:
            landmark[0] = landmark[0] *  depth_img.shape[1]
            landmark[1] = landmark[1] *  depth_img.shape[0]
        for landmark in landmarks:
            cv2.circle(depth_img, (landmark[1],landmark[0]), 0.2, 250, 0.5)
        return depth_img

    