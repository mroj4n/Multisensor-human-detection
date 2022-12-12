import numpy as np
import cv2
import math
from typing import Tuple, Union
from sklearn.linear_model import RANSACRegressor
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
    
    
    def is_flat(self,landmarks, depth_map, threshold=35000):
        points=[]
        depths=[]
        image_rows, image_cols = depth_map.shape
        print(depth_map.shape)
        # for i in range(landmarks[24][0],landmarks[23][0]):
        #     for j in range (landmarks[12][1],landmarks[24][1]):
        #         if (i>image_rows-1 or i <0 or j>image_cols-1 or j<0 ):
        #             return False
        for i in range(int(landmarks[24][0]),int(landmarks[23][0])):
            for j in range (int(landmarks[12][1]),int(landmarks[24][1])):
                    points.append([i,j])
                    depths.append([float(depth_map[j][i].astype(float)*self.depth_scale*1000)])#distance in cm
        depths = np.array(depths)
        points = np.array(points)
        # Fit a plane to the points using the RANSAC algorithm
        ransac = RANSACRegressor(min_samples=3)
        model = ransac.fit(points, depths)

        predicted_depths = model.predict(points)
        # Evaluate the quality of the fit by calculating the sum of squared residuals
        residuals = depths - predicted_depths
        ssr = np.sum(residuals)
        ssr=abs(int(ssr))
        print(ssr)
        return ssr < threshold

    
    def get_notso_stddev(self,arr):
        # Calculate the mean of the array
        mean = np.mean(arr)

        # Calculate the sum of the squared differences from the mean
        sum_sq_diff = np.sum((abs(arr - mean))*(arr - mean))
        #sum_sq_diff = Σ((arr_i - mean)^2)
        # Calculate the standard deviation
        stddev = abs(sum_sq_diff / (len(arr) - 1))

        return stddev
    
    def doesChestHaveDepth(self,pixLandmarks,depth_img):
        x1, y1, *_ = pixLandmarks[24]
        x2, y2, *_ = pixLandmarks[23]
        x3, y3, *_ = pixLandmarks[12]
        x4, y4, *_ = pixLandmarks[24]
        depthsInChestList=[]
        image_rows, image_cols = depth_img.shape
        for i in range(x1, x2):
            for j in range(y3, y4):
                if (i>image_rows-1 or i <0 or j>image_cols-1 or j<0 ):
                    return False
    
        for i in range(x1, x2):
            for j in range(y3, y4):
                depthsInChestList.append([float(depth_img[i][j].astype(float)*self.depth_scale*1000)])#distance in cm
        
        if self.is_flat(depthsInChestList):
            print('The object is flat')
        else:
            print('The object is not flat')
        return 0


    # def doesChestHaveDepth(self,pixLandmarks,depth_img):
    #     depthsInChestList=[]
    #     image_rows, image_cols = depth_img.shape
    #     for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
    #         for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
    #             if (i>image_rows-1 or i <0 or j>image_cols-1 or j<0 ):
    #                 return False
    
    #     for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
    #         for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
    #             if (int(depth_img[i][j])!=0):
    #                 depthsInChestList.append(depth_img[i][j].astype(float)*self.depth_scale*1000)#distance in cm
    #     depthsInChest=np.array(depthsInChestList)
    #     print(np.std(depthsInChest))
    #     avg= np.mean(depthsInChest)
    #     flat_factor = abs(np.sum((depthsInChest-avg)*abs(depthsInChest-avg))/len(depthsInChest))
    #     ##Should be tested with a flaat
    #     print("chest flat_factor",flat_factor)
    #     return flat_factor > 1e-4
    
    def faceDepth(self,pixLandmarks,depth_img):
        depthsF=[]
        image_rows, image_cols = depth_img.shape
        for i in range(pixLandmarks[6][0],pixLandmarks[3][0]):
            for j in range (pixLandmarks[6][1],pixLandmarks[10][1]):
                if (i>image_rows-1 or i <0 or j>image_cols-1 or j<0 ):
                    return False

        for i in range(pixLandmarks[6][0],pixLandmarks[3][0]):
            for j in range (pixLandmarks[6][1],pixLandmarks[10][1]):
                if (int(depth_img[i][j])!=0):
                    depthsF.append(depth_img[i][j].astype(float)*self.depth_scale*1000)#distance in cm
        depths=np.array(depthsF)
        avg= np.mean(depths)
        flat_factor = abs(np.sum((depths-avg)*abs(depths-avg))/len(depths))
        ##Should be tested with a flaat
        print("FACE flat_factor",flat_factor)
        return flat_factor > 1e-2

    def ThermalDetection(self,pixLandmarks,depth_img,thermalVals,minTemp,maxTemp):
        TempOfStomach=0
        iters=0
        thermalconfidence=0
        image_rows, image_cols = depth_img.shape
        for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
            for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
                if (i>image_rows-1 or i <0 or j>image_cols-1 or j<0 ):
                    return 0
        for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
            for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
                if (int(depth_img[i][j])!=0):
                    TempOfStomach=TempOfStomach+thermalVals[int(i/80)][int(j/60)]
                    iters=iters+1
        if(iters>0):
            TempOfStomach=TempOfStomach/iters
            if (TempOfStomach>29 and TempOfStomach < 38):
                thermalconfidence=thermalconfidence+5
            if ((((TempOfStomach/maxTemp[0])*100) >10) or (((TempOfStomach/minTemp[0])*100)>110) ):
                thermalconfidence=thermalconfidence+1
        return thermalconfidence



    def detect(self,landmarks,depth_img,thermalVals,minTemp,maxTemp):
        pixLandmarks=self.getPixelLandmarks(landmarks,depth_img)

        DepthConfidence=0
    
        if (pixLandmarks[24][2] and pixLandmarks[23][2] and pixLandmarks[12][2] and pixLandmarks[11][2]):
            #checks if all 4 points for chest is existing
            if self.is_flat(pixLandmarks,depth_img):
                print('The object is flat')
            else:
                print('The object is not flat')
        #if(faceDetected):
            if(self.faceDepth(pixLandmarks,depth_img)):
                DepthConfidence=DepthConfidence+3

            print("ThermalConfidence",  self.ThermalDetection(pixLandmarks,depth_img,thermalVals,minTemp,maxTemp))
            print("DepthConfidence",DepthConfidence)

        for landmark in pixLandmarks:
            cv2.circle(depth_img, (landmark[0],landmark[1]), 5, 250, 2)
        return depth_img
    
