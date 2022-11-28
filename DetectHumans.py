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

from DepthDetector import DepthDetector

ge = GridEye()
grideye_image = ge.GridValueOpenCVFormat()

reals = RealSense()
color_image, depth_colormap, depth_scale = reals.getImage()

detector = ptm.poseDetector()
depthDetector=DepthDetector()
# yolo= YOLOdetector()
while True:
    color_image, depth_colormap, depth_scale = reals.getImage()
    grideye_image = ge.GridValueOpenCVFormat()
    detectPoseRGB, detectPoseDepth, landmarks = detector.findPoseAndDrawLandmarks(
        color_image, depth_colormap)
    # yoloDetect = yolo.predict(color_image)
    #cv2.imshow("Spark image", grideye_image)
    cv2.imshow("Color RealSense image", color_image)
    cv2.imshow("Depth RealSense image", detectPoseDepth)

    #cv2.imshow("depth_frame", depth_scale)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
