A detector that detects a human and then checks if the detected object's depth and temperature to confirm if it is a human or not.


# Try it out
Some sample data is provided in `images/`, it can be tested by running `images.py`

### Setup 
#### Download weights in `yoloV3/`  `wget https://pjreddie.com/media/files/yolov3.weights`

#  Retrieving Data
The code for retrieving data was written in Python. Multiple libraries were used to
retrieve the data and change the data type to fit the project. The data was collected and
saved in Jetson Nano and then later transferred to an external computer.
`recorder.py` is used to save the data
## Realsense
In order to retrieve data from Realsense the pyRealSense package was used. It's implementation is in `realsense_data.py`
The RGB data had a resolution of 640x480. Although the
depth sensor had a higher resolution, it was cropped to have the same resolution during
the alignment phase.
The RGB values were stored as an image. The Depth value and the Depth scale were
stored as .npy files as saving the Depth values as image led to the loss of data.


##  Grid-EYE AMG8833
The Grid-EYE AMG8833 uses I2C connection in order to retrieve the data. With I2C
values for each row can be read. The values were read and transformed into a NumPy
array. And then combined in the form of a matrix. This gave the temperatures for each
pixel in the form of a matrix. This was represented as an image and also saved as a .npy
file. The resolution of the data from the thermal sensor is an 8x8 matrix.
It's implementation is in `sensor_data.py`


# Data processing – Detecting human
##  RGB Data processing
The RGB data was stored as an image. This image was put through pre-trained models for human detection. 
The two models work together to detect multiple humans. This is a type of multi-stage system just for detecting humans in RGB images. 
 
The idea was to use mediapipe to detect humans. MediaPipe is a pre-trained human recognizing model by google. 
It is quite powerful as it can detect points on the human body. Thus, it can detect "pose" of a human. 
However, it only works with one person at a time. 
To solve this YOLO was used.YOLO (You Only Look Once) is a popular object detection algorithm that is fast and accurate. 
YOLO returns the area where the human is detected, then all the region where human was detected is sent to mediapipe. 
Then mediapipe gives back all the points of the human for each region. Thus, in the end, the points for all humans are fetched. 

see `yoloV3.py` and `PoseTrackingModule.py`




##  Depth Data processing
Here, the idea was to map the points or the "landmarks" received at RGB Data processing mapped to the depth map first.  
The least squares method was used to fit a plane to the points in the depth map and then measure the sum of the squared residuals.  
A small value for the RSS would indicate that the points lie close to the fitted plane, and therefore the object is likely to be flat. 
 
Implementated at `DepthAndThermalDetector.py`.


## Thermal Data processing
Similar to Depth Data processing, in Thermal Data processing, the landmarks for the human is mapped on top of the thermal image.
Then it is checked if the human has a temperature of around 36C. 
It is also checked if the human is around 10% different from its environment. As in, we try to check if there is any anomaly where the human is detected.
Both these checks contribute to the confidence of the detection by the depth camera however the degree checker has more weight in the decision. 
Implementated at `DepthAndThermalDetector.py::ThermalDetection()`
