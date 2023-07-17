import cv2
from utils.common import *
import numpy
import time
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserverWithCallbacks

def detect_water_see(image):
     blue_lower = numpy.array([111.35, 163.3 , 158.7 ])
     blue_upper = numpy.array([121.84722222, 211.55, 245.])
     see_low = numpy.array([106, 155, 234], dtype=numpy.uint8)
     see_high = numpy.array([126, 175, 254], dtype=numpy.uint8)
 
     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     mask = cv2.inRange(hsv, blue_lower, blue_upper)
     mask_see = cv2.inRange(hsv, see_low, see_high)
     return mask, mask_see


def main():
    mc = MCConnector.connect(name='Cristina', video=True)
    rob = RobustObserverWithCallbacks(mc)
    while True:
        rob.updateAllObservations()
        time.sleep(0.15)
        image = rob.getCachedObserve('getImageFrame')
        if image is not None:
            image = image.pixels
        else:
            continue

        mask, mask_see = detect_water_see(image)
        cv2.imshow('img', image)
        cv2.imshow('water', mask)
        cv2.imshow('see', mask_see)
        cv2.waitKey(300)

if __name__ == '__main__':
   main()

