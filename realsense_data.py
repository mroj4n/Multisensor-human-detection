import pyrealsense2 as rs
import numpy as np
import cv2

class RealSense():
    def __init__(self):
        # Configure streams
        self.pipeline=rs.pipeline()
        self.config=rs.config()
        
        self.pipeline_profile=self.config.resolve(rs.pipeline_wrapper(self.pipeline))
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            print("L500 detected")
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.profile=self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        self.clipping_distance_in_meters = 1 #1 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def getDepthScale(self):
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        return depth_scale

    def getImage(self):
        frames = self.pipeline.wait_for_frames()

        # alighn frames
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        colorizer = rs.colorizer()
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image
