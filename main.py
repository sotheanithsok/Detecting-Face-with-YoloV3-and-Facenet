import os
import math
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from PIL import Image

from imageFinder import ImageFinder
finder = ImageFinder()

for group in range(1,8,1):
    precision = []
    recall = []
    images = []
    accuracy =[]
    metrics = []

    for t in range(20,200,20):
        print("-----------------------------------")
        start = time.time()
        finder.find_images(group,threshold= t/100.)
        end = time.time()
        print("Group: ", group)
        print("Threshold: ", t/100.)
        print("Execution time: ", (end - start))
        print("Images: ", len(finder.get_detected_images()))
        print("Recall: ", finder.get_recall())
        print("Precision: ", finder.get_precision())
        print("Accuracy:", finder.get_accuracy())
        print("Metrics: (%i, %i, %i, %i)" %(finder.tp,finder.tn,finder.fp,finder.fn))
        images.append( len(finder.get_detected_images()))
        precision.append(finder.get_precision())
        recall.append(finder.get_recall())
        accuracy.append(finder.get_accuracy())
        metrics.append((finder.tp,finder.tn,finder.fp,finder.fn))
        print("-----------------------------------")
    
    print("-----------------------------------")
    print("Images:", images)
    print("Recall:", recall)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Metrics", metrics)
    print("-----------------------------------")   
