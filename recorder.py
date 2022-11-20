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
grideye_image = ge.GridValueOpenCVFormat()

reals = RealSense()
color_image, depth_colormap = reals.getImage()

Spark_filename= "images/Spark_image/"
Color_filename= "images/Color_RealSense/"
Depth_filename= "images/Depth_RealSense/"
file_ext = ".jpg"
counter=0

shutil.rmtree("images")
os.mkdir("images")
os.mkdir("images/Spark_image/")
os.mkdir("images/Color_RealSense/")
os.mkdir("images/Depth_RealSense/")

while True:
    grideye_image = ge.GridValueOpenCVFormat()
    color_image, depth_colormap = reals.getImage()

    cv2.imwrite(Spark_filename+str(counter)+file_ext, grideye_image)
    cv2.imwrite(Color_filename+str(counter)+file_ext, color_image)
    cv2.imwrite(Depth_filename+str(counter)+file_ext, depth_colormap)
    cv2.imshow("RealSense image", color_image)
    counter=counter+1
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
