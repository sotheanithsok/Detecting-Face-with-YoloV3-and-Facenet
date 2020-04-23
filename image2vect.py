import numpy as np
import facenet 
import yolo

class ImageVectorize:
    def __init__(self, yolo_model = yolo.load_model(), facenet_model = facenet.load_model()):
        self._yolo_model = yolo_model
        self._facenet_model = facenet_model

    def image2vect(self, image):
        # Get metrics from yolo detected image
        out_scores, out_boxes, out_classes = self._yolo_model.detect_image(image)
        self._yolo_model.close_session()
        # If box is not detected, return 128 zeros
        # Out_box.shape expected to be (1,4). One tuple of four items (top, left, bottom, right)
        if(out_boxes.shape == (1,4)):
            imageVector = self._cropImage(out_boxes, image)
        else:
            imageVector = np.zeros((128,))

        return imageVector

    def _cropImage(self, out_boxes, image):
        # Get box size
        top, left, bottom, right = out_boxes[0] 

        croppedImage = image.crop((left, top, right, bottom)).resize((160,160))  

        # Reshape to np.array for color channels
        croppedImage_array = np.array(list(croppedImage.getdata())).reshape((160,160,3)) 
        imageVector = self._facenet_model.predict(np.array([croppedImage_array])).flatten()

        # If output shape does not match expected output, return all zeros
        # Expected shape of vector is one tuple of 128 values
        if(imageVector.shape != (128,)):
            imageVector = np.zeros((128,)) 

        return imageVector
