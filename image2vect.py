import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import facenet 
import yolo
from sklearn.preprocessing import normalize, StandardScaler

class ImageVectorize:
    def __init__(self, yolo_model = None, facenet_model = None):
        if yolo_model==None:
            self._yolo_model=yolo.load_model()
        else:
            self._yolo_model = yolo_model

        if facenet_model==None:
            self._facenet_model=facenet.load_model()
        else:
            self._facenet_model = facenet_model

    def image2vect(self, image):
        """
        Map the highest probability face onto 128 vectors representation
        """

        # Get metrics from yolo detected image
        out_scores, out_boxes, out_classes = self._yolo_model.detect_image(image)

        # If box is not detected, return 128 zeros
        # Out_box.shape expected to be (1,4). One tuple of four items (top, left, bottom, right)
        if(out_boxes.size!=0):
            imageVector = self._cropImage(out_boxes, image)
        else:
            imageVector = np.full((1,128), np.inf)

        return imageVector

    def _cropImage(self, out_boxes, image):
        """
        Crop face out of an image and map it onto 128 vectors representation
        """
        images_array = []
        for out_box in out_boxes:
            # Get box size
            top, left, bottom, right = out_box

            #Check for boundaries
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            #Crop and resize image
            croppedImage = image.crop((left, top, right, bottom)).resize((160,160))  

            # Reshape to np.array for color channels
            croppedImage_array = np.array(list(croppedImage.getdata())).reshape((160,160,3)) 
            images_array.append(croppedImage_array)
            break
        
        #Preprocessing
        images_array = np.array(images_array, dtype="float32")
        images_array /=255.0
        images_array -=0.5 

        #Feed cropped images into facenet
        imageVector = self._facenet_model.predict(images_array)

        #Normalize the vector
        imageVector = normalize(imageVector, norm='l2')

        return imageVector
