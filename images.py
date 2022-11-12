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
import HandTrackingModule as htm
import glob
import os

from sensordata import GridEye
from realsense_data import RealSense

     

ge= GridEye()
grideye_image=ge.GridValueOpenCVFormat()

reals=RealSense()
color_image, depth_colormap=reals.getImage()

detector = htm.handDetector()
detected_hands=color_image
while True:
    grideye_image=ge.GridValueOpenCVFormat()
    color_image, depth_colormap=reals.getImage()
    detected_hands = detector.findHands(color_image, draw=True)
    depth_Det = detector.findHands(depth_colormap, draw=True)
    #cv2.imshow("Spark image", grideye_image)
    #cv2.imshow("Color RealSense image", color_image)
    cv2.imshow("Depth RealSense image", depth_Det)
    cv2.imshow("Dete RealSense image", detected_hands)
    k=cv2.waitKey(1)
    if k==27:
        cv2.destroyAllWindows()
        break
