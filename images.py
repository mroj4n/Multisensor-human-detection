import numpy as np
import cv2


import mediapipe as mp

import PoseTrackingModule as ptm
import os

from yoloV3 import YOLOdetector

from DepthDetector import DepthDetector


main_folder_name="Recordings/mixes_copy/"

Spark_filename= main_folder_name+"Spark_image/"
Spark_numpys=main_folder_name+"Spark_npys/"
Spark_mins=main_folder_name+"Spark_mins/"
Spark_maxs=main_folder_name+"Spark_maxs/"

Color_filename= main_folder_name+"Color_RealSense/"
Depth_filename= main_folder_name+"Depth_RealSense/"
Depthnp_filename = main_folder_name+"Depthnp_RealSense/"

file_ext = ".jpg"


Color_RealSense= []
Depth_RealSense= []
Depthnp_RealSense= []
Spark_image= []
Spark_maxS= []
Spark_minS= []
Spark_npys= []


for file in os.listdir(Color_filename):
    Color_RealSense.append(cv2.imread(Color_filename+file))

for file in os.listdir(Depth_filename):
    Depth_RealSense.append(cv2.imread(Depth_filename+file))

for file in os.listdir(Depthnp_filename):
    Depthnp_RealSense.append(np.load(Depthnp_filename+file))

for file in os.listdir(Spark_filename):
    Spark_image.append(cv2.imread(Spark_filename+file))

for file in os.listdir(Spark_maxs):
    Spark_maxS.append(np.load(Spark_maxs+file))

for file in os.listdir(Spark_mins):
    Spark_minS.append(np.load(Spark_mins+file))

for file in os.listdir(Spark_numpys):
    Spark_npys.append(np.load(Spark_numpys+file))


depth_scale=np.load(main_folder_name+'depth_scale.npy')

detector = ptm.poseDetector()
depthDetector=DepthDetector(depth_scale=depth_scale)
yolo= YOLOdetector()



##mediapipe only
for i in range(len(Color_RealSense)):
    grideye_image = Spark_image[i]
    color_image  = Color_RealSense[i]
    depth_colormap = Depth_RealSense[i]
    grideye_values=Spark_npys[i]
    minTemp=Spark_minS[i]
    maxTemp=Spark_maxS[i]
    depth_map=Depthnp_RealSense[i]



    detectPoseRGB, detectPoseDepth, landmarks = detector.findPoseAndDrawLandmarks(
    color_image, depth_map)

    
    if(landmarks):
        depth_map=depthDetector.detect(landmarks,depth_map,grideye_values,minTemp,maxTemp)
    
    
    cv2.imshow("Color RealSense image", color_image)
    print(i)
    print()
    print()
    print()
    #cv2.imshow("Depth RealSense image", detectPoseDepth)
    #cv2.imshow("Dete RealSense image", detectPoseRGB)
    k =cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break


