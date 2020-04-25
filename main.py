import os
import math
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from PIL import Image

from imageFinder import ImageFinder
finder = ImageFinder()


for group in range(1,8,1):
    for t in range(0,105,5):
        start = time.time()
        finder.find_images(group,threshold= t/100.)
        end = time.time()
        print("Group: ", group)
        print("Threshold: ", t/100.)
        print("Execution time: ", (end - start))
        print("Images: ", len(finder.get_detected_images()))
        print("Precision: ", finder.get_precision())
        print("Recall: ", finder.get_recall())
        print("-----------------------------------")
