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
color_image, depth_map = reals.getImage()
depth_scale=reals.getDepthScale()

detector = ptm.poseDetector()
depthDetector=DepthDetector()
# yolo= YOLOdetector()
while True:
    color_image, depth_map = reals.getImage()
    grideye_image = ge.GridValueOpenCVFormat()
    detectPoseRGB, detectPoseDepth, landmarks = detector.findPoseAndDrawLandmarks(
        color_image, depth_map)
    # yoloDetect = yolo.predict(color_image)
    #cv2.imshow("Spark image", grideye_image)

    distanceCM=depth_map[205,305].astype(float)*depth_scale*100
    cv2.circle(color_image,(305,205),4,(0,0,0))
    cv2.putText(color_image,"{}cm".format(distanceCM),(305,195),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    
    cv2.imshow("Color RealSense image", color_image)
    
    

    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
