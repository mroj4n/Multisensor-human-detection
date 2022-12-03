import numpy as np
import cv2
import math
from typing import Tuple, Union
class DepthDetector():
    def __init__(self,depth_scale):
        self.depth_scale=depth_scale
    
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
        image_rows, image_cols = depth_img.shape
        pixLandmarks=[]
        if landmarks:
            for landmark in landmarks:
                if (landmark[0] and landmark[1]):
                    x_px,y_px,isLandmarkVisible = self._normalized_to_pixel_coordinates(landmark[0],landmark[1],image_cols, image_rows)
                pixLandmarks.append([x_px,y_px,isLandmarkVisible])
        return pixLandmarks
    
    def doesChestHaveDepth(self,pixLandmarks,depth_img):
        depthsInChestList=[]
        image_rows, image_cols = depth_img.shape
        for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
            for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
                if (i>image_rows or i <0 or j>image_cols or j<0 ):
                    return False

        for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
            for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
                if (int(depth_img[i][j])!=0):
                    depthsInChestList.append(depth_img[i][j].astype(float)*self.depth_scale*1000)#distance in cm
        depthsInChest=np.array(depthsInChestList)
        avg= np.mean(depthsInChest)
        flat_factor = abs(np.sum((depthsInChest-avg)*abs(depthsInChest-avg))/len(depthsInChest))
        ##Should be tested with a flaat
        ## print("chest")
        ## print (flat_factor)
        return flat_factor > 1e-4
    
    def faceDepth(self,pixLandmarks,depth_img):
        depthsF=[]
        image_rows, image_cols = depth_img.shape
        for i in range(pixLandmarks[6][0],pixLandmarks[3][0]):
            for j in range (pixLandmarks[6][1],pixLandmarks[10][1]):
                if (i>image_rows or i <0 or j>image_cols or j<0 ):
                    return False

        for i in range(pixLandmarks[6][0],pixLandmarks[3][0]):
            for j in range (pixLandmarks[6][1],pixLandmarks[10][1]):
                if (int(depth_img[i][j])!=0):
                    depthsF.append(depth_img[i][j].astype(float)*self.depth_scale*1000)#distance in cm
        depths=np.array(depthsF)
        avg= np.mean(depths)
        flat_factor = abs(np.sum((depths-avg)*abs(depths-avg))/len(depths))
        ##Should be tested with a flaat
        ## print("FACE")
        ## print (flat_factor)
        return flat_factor > 1e-2


    def detect(self,landmarks,depth_img):
        pixLandmarks=self.getPixelLandmarks(landmarks,depth_img)

        DepthConfidence=0
    
        if (pixLandmarks[24][2] and pixLandmarks[23][2] and pixLandmarks[12][2] and pixLandmarks[11][2]):
            #checks if all 4 points for chest is existing
            if(self.doesChestHaveDepth(pixLandmarks,depth_img)):
                DepthConfidence=DepthConfidence+10
        #if(faceDetected):
            if(self.faceDepth(pixLandmarks,depth_img)):
                DepthConfidence=DepthConfidence+3
        print(DepthConfidence)
        for landmark in pixLandmarks:
            cv2.circle(depth_img, (landmark[0],landmark[1]), 5, 250, 2)
        return depth_img
    
