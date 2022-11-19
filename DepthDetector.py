import numpy as np
import cv2

class DepthDetector():
    def __init__():
        pass
    def detect(self,landmarks,depth_img):
        denormalize = normalized_d * (max_d - min_d) + min_d
        for landmark in landmarks:
            landmark[0] = landmark[0] *  depth_img.shape[1]
            landmark[1] = landmark[1] *  depth_img.shape[0]