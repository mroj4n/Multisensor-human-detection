try:
    import smbus2 as smbus
except (ImportError):
    import smbus
import numpy as np
import cv2
from time import sleep
from pprint import pprint
import pyrealsense2 as rs

import mediapipe as mp
import time
import PoseTrackingModule as ptm
import glob
import os

from sensordata import GridEye
from realsense_data import RealSense
from yoloV3 import YOLOdetector
     

ge= GridEye()
grideye_image=ge.GridValueOpenCVFormat()

reals=RealSense()
color_image, depth_colormap=reals.getImage()

detector = ptm.poseDetector()
detectPose=color_image
detectPoseDepth=depth_colormap
# yolo= YOLOdetector()
while True:
    grideye_image=ge.GridValueOpenCVFormat()
    color_image, depth_colormap=reals.getImage()
    detectPoseRGB , detectPoseDepth  = detector.findpose(color_image, depth_colormap)
    # yoloDetect = yolo.predict(color_image)
    cv2.imshow("Spark image", grideye_image)
    cv2.imshow("Color RealSense image", color_image)
    cv2.imshow("Depth RealSense image", detectPoseDepth)
    cv2.imshow("Dete RealSense image", detectPoseRGB)
    k=cv2.waitKey(1)
    if k==27:
        cv2.destroyAllWindows()
        break
