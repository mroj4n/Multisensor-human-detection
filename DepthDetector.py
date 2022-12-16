import numpy as np
import cv2

from scipy.ndimage.filters import median_filter
class DepthDetector():
    def __init__(self,depth_scale):
        self.depth_scale=depth_scale
    

    def is_flat(self,x1,x2,y1,y2, depth_map, threshold=60000):

        filtered_depth_map = depth_map
        points=[]
        depths=[]
        for i in range(x1,x2):
            for j in range (y1,y2):    
                points.append([float(filtered_depth_map[j][i].astype(float)*self.depth_scale*1000)])#distance in cm
            depths.append(points)
            points=[]
        depths = np.array(depths)
        # Compute the centroid of the points
        centroid = np.mean(depths, axis=0)
        
        # Subtract the centroid from the points to center them at the origin
        centered_points = depths - centroid
        
        # Use singular value decomposition to fit a plane to the centered points
        U, S, V = np.linalg.svd(centered_points)
        
        # The normal vector of the fitted plane is the last column of V
        normal = V[-1, :]

        # Compute the sum of squared residuals
        residuals = np.sum((centered_points @ normal)**2)

        return residuals<threshold

    
    # def get_notso_stddev(self,arr):
    #     # Calculate the mean of the array
    #     mean = np.mean(arr)

    #     # Calculate the sum of the squared differences from the mean
    #     sum_sq_diff = np.sum((abs(arr - mean))*(arr - mean))
    #     #sum_sq_diff = Σ((arr_i - mean)^2)
    #     # Calculate the standard deviation
    #     stddev = abs(sum_sq_diff / (len(arr) - 1))

    #     return stddev

    def ThermalDetection(self,pixLandmarks,depth_img,thermalVals,minTemp,maxTemp):
        image_rows, image_cols = depth_img.shape
        for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
            for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
                if (i>image_rows-1 or i <0 or j>image_cols-1 or j<0 ):
                    return 0
        TempOfStomach=0
        iters=0
        thermalconfidence=0
        for i in range(pixLandmarks[24][0],pixLandmarks[23][0]):
            for j in range (pixLandmarks[12][1],pixLandmarks[24][1]):
                if (int(depth_img[i][j])!=0):
                    TempOfStomach=TempOfStomach+thermalVals[int(i/80)][int(j/60)]
                    iters=iters+1
        if(iters>0):
            TempOfStomach=TempOfStomach/iters

        TempOfFace=0
        iters=0
        for i in range(pixLandmarks[6][0],pixLandmarks[3][0]):
            for j in range (pixLandmarks[6][1],pixLandmarks[10][1]):
                if (int(depth_img[i][j])!=0):
                    TempOfFace=TempOfFace+thermalVals[int(i/80)][int(j/60)]
                    iters=iters+1
        if(iters>0):
            TempOfFace=TempOfFace/iters
            if (TempOfFace>29 and TempOfFace < 38):
                thermalconfidence=thermalconfidence+5
            if ((((TempOfFace/maxTemp[0])*100) >30) or (((TempOfFace/minTemp[0])*100)>130) ):
                thermalconfidence=thermalconfidence+5
        
            
        if( (TempOfFace/TempOfStomach)*100 ) > 10 :
            thermalconfidence=thermalconfidence+10

        if (TempOfStomach>29 and TempOfStomach < 38):
            thermalconfidence=thermalconfidence+6
        if ((((TempOfStomach/maxTemp[0])*100) >30) or (((TempOfStomach/minTemp[0])*100)>130) ):
            thermalconfidence=thermalconfidence+5
        thermalconfidence=int((thermalconfidence/24)*100)
        return thermalconfidence

    def draw_prediction(self, img, x, y, x_plus_w, y_plus_h,depthConfidence,thermalconfidence):
        
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0,0,0), 2)
        thickness=1
        font_size=0.4
        color=(255,0,255)
        cv2.putText(img, "depthConfidence:"+str(depthConfidence)+"%", (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        cv2.putText(img, "thermalconfidence:"+str(thermalconfidence)+"%", (x-25, y-25),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
    def depthDetection(self,pixLandmarks,depth_img):
        DepthConfidence=0
        #chest detection
        if (pixLandmarks[24][2] and pixLandmarks[23][2] and pixLandmarks[12][2] and pixLandmarks[11][2]):
            #checks if all 4 points for chest is existing
            if not (self.is_flat(pixLandmarks[24][0],pixLandmarks[23][0],pixLandmarks[12][1],pixLandmarks[24][1], depth_img, threshold=10000000)):
                DepthConfidence=DepthConfidence+6
        #faceDetection
        if (pixLandmarks[6][2] and pixLandmarks[3][2] and pixLandmarks[10][2] and pixLandmarks[9][2]):
            if not (self.is_flat(pixLandmarks[6][0],pixLandmarks[3][0],pixLandmarks[6][1],pixLandmarks[10][1], depth_img, threshold=100000)):
                DepthConfidence=DepthConfidence+3
        DepthConfidence=int((DepthConfidence/9)*100)
        return DepthConfidence
    
    def detect(self,pixLandmarks,depth_img,color_image,thermalVals,minTemp,maxTemp):
        depthConfidence=self.depthDetection(pixLandmarks,depth_img)
        thermalconfidence=self.ThermalDetection(pixLandmarks,depth_img,thermalVals,minTemp,maxTemp)
        print("ThermalConfidence",  thermalconfidence)
        print("DepthConfidence",depthConfidence)

        #draw landmarks on depth_img
        # for landmark in pixLandmarks:
        #     cv2.circle(depth_img, (landmark[0],landmark[1]), 5, 250, 2)
        return depth_img,depthConfidence,thermalconfidence
    
    def detectWithYOLO(self,pixLandmarks,depth_img,color_image,thermalVals,minTemp,maxTemp,x, y, x_plus_w, y_plus_h):
        depthConfidence=self.depthDetection(pixLandmarks,depth_img)
        thermalconfidence=self.ThermalDetection(pixLandmarks,depth_img,thermalVals,minTemp,maxTemp)

        self.draw_prediction(color_image, x, y, x_plus_w, y_plus_h,depthConfidence,thermalconfidence)

        return depth_img,color_image,depthConfidence,thermalconfidence
    
