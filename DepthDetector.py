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
        return None
      x_px = min(math.floor(normalized_x * image_width), image_width - 1)
      y_px = min(math.floor(normalized_y * image_height), image_height - 1)
      return x_px, y_px
    
    def detect(self,landmarks,depth_img):
        image_rows, image_cols, _ = depth_img.shape
        pixLandmarks=[]
        if landmarks:
            for landmark in landmarks:
                x_px,y_px = self._normalized_to_pixel_coordinates(landmark[0],landmark[1],image_cols, image_rows)
                pixLandmarks.append([x_px,y_px])
            for landmark in pixLandmarks:
                cv2.circle(depth_img, (landmark[0],landmark[1]), 5, 250, 2)
        return depth_img
    