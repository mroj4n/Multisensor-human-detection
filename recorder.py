try:
    import smbus2 as smbus
except (ImportError):
    import smbus
import shutil
import numpy as np
import cv2
from time import sleep
from pprint import pprint
import pyrealsense2 as rs


import time
import glob
import os

from sensordata import GridEye
from realsense_data import RealSense



ge = GridEye()
grideye_image = ge.GetGridValue()

reals = RealSense()
color_image, depth_map = reals.getImage()
depth_scale=reals.getDepthScale()

main_folder_name="recordings"
i=0
main_folder_name=main_folder_name+str(i)
while os.path.exists(main_folder_name+"/"):
    i=i+1
    main_folder_name=main_folder_name[:-1]+str(i)
main_folder_name=main_folder_name+"/"
Spark_filename= main_folder_name+"Spark_image/"
Spark_numpys=main_folder_name+"Spark_npys/"
Spark_mins=main_folder_name+"Spark_mins/"
Spark_maxs=main_folder_name+"Spark_maxs/"

Color_filename= main_folder_name+"Color_RealSense/"
Depth_filename= main_folder_name+"Depth_RealSense/"
Depthnp_filename = main_folder_name+"Depthnp_RealSense/"
file_ext = ".jpg"
counter=0

if os.path.exists(main_folder_name):
    shutil.rmtree(main_folder_name)
os.mkdir(main_folder_name)
os.mkdir(Spark_filename)
os.mkdir(Spark_numpys)
os.mkdir(Spark_mins)
os.mkdir(Spark_maxs)
os.mkdir(Color_filename)
os.mkdir(Depth_filename)
os.mkdir(Depthnp_filename)
np.save(main_folder_name+"depth_scale.npy",depth_scale)
while True:
    grideye_image = ge.GetGridValue()
    color_image, depth_map = reals.getImage()
    grideye_values,minTemp,maxTemp=ge.GetGridValue(ImageMode=False)
    
    np.save(Spark_numpys+str(counter)+".npy",grideye_values)
    np.save(Spark_mins+str(counter)+".npy",minTemp)
    np.save(Spark_maxs+str(counter)+".npy",maxTemp)

    cv2.imwrite(Spark_filename+str(counter)+file_ext, grideye_image)
    cv2.imwrite(Color_filename+str(counter)+file_ext, color_image)
    cv2.imwrite(Depth_filename+str(counter)+file_ext, depth_map)
    np.save(Depthnp_filename+str(counter)+".npy",depth_map)
    cv2.imshow("RealSense image", depth_map)
    counter=counter+1
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
