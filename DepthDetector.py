import numpy as np
import cv2
import math
from typing import Tuple, Union
class DepthDetector():
    def __init__(self):
        pass
    
    def _normalized_to_pixel_coordinates(
        self,normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
      """Converts normalized value pair to pixel coordinates."""

      # Checks if the float value is between 0 and 1.
      def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

      if not (is_valid_normalized_value(normalized_x) and
              is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px,y_px, False
      x_px = min(math.floor(normalized_x * image_width), image_width - 1)
      y_px = min(math.floor(normalized_y * image_height), image_height - 1)
      return x_px, y_px , True
    
    def getPixelLandmarks(self,landmarks,depth_img):
        image_rows, image_cols, _ = depth_img.shape
        pixLandmarks=[]
        if landmarks:
            for landmark in landmarks:
                if (landmark[0] and landmark[1]):
                    x_px,y_px,isLandmarkVisible = self._normalized_to_pixel_coordinates(landmark[0],landmark[1],image_cols, image_rows)
                pixLandmarks.append([x_px,y_px,isLandmarkVisible])
        return pixLandmarks
   
    def doesChestHaveDepth(self,pixLandmarks,depth_img):
        depthsInChest=[]
        for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
            for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
                depthsInChest.append(int(depth_img[i][j][0])+int(depth_img[i][j][1])+int(depth_img[i][j][2]))
        if (np.std(depthsInChest)<10):
            return False
        return True
    
    def faceDepth(self,landmarks,depth_img):
        depths=[]
        anti=0

        for i in range(0,10):
            if (landmarks[i][2]):
                depths.append(int(depth_img[landmarks[i][0]][landmarks[i][1]][0])+int(depth_img[landmarks[i][0]][landmarks[i][1]][1])+int(depth_img[landmarks[i][0]][landmarks[i][1]][2]))
                anti=anti+1
        if (np.std(depths)<10):
            return False
        return True




    def detect(self,landmarks,depth_img):
        image_rows, image_cols, _ = depth_img.shape
        pixLandmarks=self.getPixelLandmarks(landmarks,depth_img)

        DepthConfidence=0
    
        if (pixLandmarks[24][2] and pixLandmarks[23][2] and pixLandmarks[12][2] and pixLandmarks[11][2]):
            #checks if all 4 points for chest is existing
            if(self.doesChestHaveDepth(pixLandmarks,depth_img)):
                DepthConfidence=DepthConfidence+1
        
        
        faceDetected = True
        for i in range(0,10):
            if (not landmarks[i][2]):
                faceDetected=False
        if(faceDetected):
            if(self.faceDepth(pixLandmarks,depth_img)):
                DepthConfidence=DepthConfidence+1
        print(DepthConfidence)
        # for landmark in pixLandmarks:
        #     cv2.circle(depth_img, (landmark[0],landmark[1]), 5, 250, 2)
        return depth_img
    