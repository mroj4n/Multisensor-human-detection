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

from DepthAndThermalDetector import DepthAndThermalDetector

ge = GridEye()
grideye_image = ge.GetGridValue()
reals = RealSense()
color_image, depth_map = reals.getImage()
depth_scale=reals.getDepthScale()

detector = ptm.poseDetector()
depthAndThermalDetector=DepthAndThermalDetector(depth_scale=depth_scale)
yolo= YOLOdetector()
while True:
    color_image, depth_map = reals.getImage()
    grideye_image = ge.GetGridValue()
    grideye_values,minTemp,maxTemp=ge.GetGridValue(ImageMode=False)
    detectPoseRGB, detectPoseDepth, landmarks = detector.findPoseAndDrawLandmarks(
        color_image, depth_map)

    
    yoloDetects = yolo.predict(color_image)
    landmarks=[]
    yoloCoods=[]
    for yoloDetect in yoloDetects:
        detectPoseRGB, detectPoseDepth, landmark = detector.findPoseAndDrawLandmarksWithYolo(
        color_image, depth_map,yoloDetect[0],yoloDetect[1],yoloDetect[2],yoloDetect[3] )
        landmarks.append(landmark)
        if(landmark):
            depth_map,color_image,_,_=depthAndThermalDetector.detectWithYOLO(landmark,depth_map,color_image,grideye_values,minTemp,maxTemp,yoloDetect[0],yoloDetect[1],yoloDetect[2],yoloDetect[3])

    cv2.imshow("Color RealSense image", color_image)
    
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
