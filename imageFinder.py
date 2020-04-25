from yolo import load_model
from data import load_data
from image2vect import ImageVectorize
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math



class ImageFinder:
    # Initialzation with loading yolo model and member list
    def __init__(self):
        self._imageVectorize = ImageVectorize()
        self._yolo_model = load_model()
        self._data = load_data()
        self._anchors = []
        self._detected_images = []
        self.tp, self.fp, self.tn, self.fn = 0,0,0,0
        
    #Find anchors
    def get_anchors(self):
        return self._anchors
    
    # Find the path of photos with highest Pc by giving group ID
    def _find_anchors(self, group_number):
        anchors=[]
        vectorized_anchors=None
        for member in self._data[group_number].keys():
            p = 0
            possible_anchor_img = None

            for img in self._data[group_number][member]:
                score,_,_ = self._yolo_model.detect_image(img)
                if(score.size!=0 and p<np.max(score) ):
                    p=np.max(score)
                    possible_anchor_img=img

            if(possible_anchor_img!=None):
                anchors.append(possible_anchor_img)
                if vectorized_anchors is None:
                    vectorized_anchors = self._imageVectorize.image2vect(possible_anchor_img)
                else:
                    vectorized_anchors = np.concatenate((vectorized_anchors, self._imageVectorize.image2vect(possible_anchor_img)))

        return anchors, vectorized_anchors


    # Triplets
    def find_images(self, group_number, threshold = 0.1):
        if(int(group_number)<1 or int(group_number)>7):
            print("Invalid group number")
            return

        self._detected_images = []
        self.tp, self.fp, self.tn, self.fn = 0,0,0,0

        group_number = str(group_number)

        self._anchors, self._vectorized_anchors=self._find_anchors(group_number)

        for group in self._data.keys():
            for member in self._data[group].keys():
                for img in self._data[group][member]:

                    distance = self._calculate_minimum_euclidean_distances(img)
                    if distance<=threshold:
                        self._detected_images.append(img)                        
                        
                    #Metrics
                    if group_number==group and distance<=threshold:
                        self.tp = self.tp + 1
                    elif group_number!=group and distance<=threshold:
                        self.fp = self.fp + 1
                    elif group_number == group and not distance<=threshold:
                        self.fn = self.fn + 1
                    elif group_number != group and not distance<=threshold:
                        self.tn = self.tn +1 

        return self._detected_images
    
    def _calculate_minimum_euclidean_distances(self, img):
        minimum_distance=1000.0
        for vectorized_anchor in self._vectorized_anchors:
            if np.inf in vectorized_anchor:
                continue

            for vectorized_img in self._imageVectorize.image2vect(img):
                if np.inf in vectorized_img:
                    continue

                distance = euclidean_distances(vectorized_anchor.reshape((1,128)), vectorized_img.reshape((1,128)))
                if distance < minimum_distance:
                    minimum_distance = distance
        return minimum_distance
    
    def get_precision (self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)
    
    def get_recall(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def get_detected_images(self):
        return self._detected_images

    def get_detected_images_as_one(self, size_per_image = (160,160)):
        images=[]
        for i in self._detected_images:
            images.append(i.resize(size_per_image))

        widths = (math.ceil(math.sqrt(len(images))))*size_per_image[0]
        heights = (math.ceil(math.sqrt(len(images))))*size_per_image[1]
        new_im = Image.new('RGB', (widths, heights))

        x_offset = 0
        y_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,y_offset))

            x_offset = x_offset+size_per_image[0]
            if x_offset>=widths:
                x_offset=0
                y_offset=y_offset+size_per_image[1]


            

    
    