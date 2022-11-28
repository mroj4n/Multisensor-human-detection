import numpy as np
import cv2
from time import sleep
from pprint import pprint


import mediapipe as mp

import PoseTrackingModule as ptm
import glob
import os

from yoloV3 import YOLOdetector

from DepthDetector import DepthDetector

main_folder_name="images/"
Spark_filename= main_folder_name+"Spark_image/"
Color_filename= main_folder_name+"Color_RealSense/"
Depth_filename= main_folder_name+"Depth_RealSense/"
file_ext = ".jpg"
counter=0

images=[]
for _ in os.listdir(Spark_filename):
    images.append([Spark_filename+str(counter)+file_ext,Color_filename+str(counter)+file_ext,Depth_filename+str(counter)+file_ext])
    counter=counter+1


detector = ptm.poseDetector()
depthDetector=DepthDetector()
yolo= YOLOdetector()
for i in range(0, 144):
    grideye_image = cv2.imread(images[i][0])
    color_image  = cv2.imread(images[i][1])
    depth_colormap = cv2.imread(images[i][2])
    # yoloDetect = yolo.predict(color_image)
    detectPoseRGB, detectPoseDepth, landmarks = detector.findPoseAndDrawLandmarks(
        color_image, depth_colormap)
    if (landmarks):
        depth_colormap=depthDetector.detect(landmarks,depth_colormap)
    cv2.imshow("Spark image", grideye_image)
    
    cv2.imshow("Color RealSense image", depth_colormap)
    
    #cv2.imshow("Depth RealSense image", detectPoseDepth)
    #cv2.imshow("Dete RealSense image", detectPoseRGB)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
