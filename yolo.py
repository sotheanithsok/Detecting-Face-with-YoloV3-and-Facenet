# Special thank to Thanh Nguyen 
# for providing source code and pretrained model
# that the following codes
# are based from. 
# https://github.com/sthanhng/yoloface

from pathlib import Path
from downloaders import download_file_from_google_drive
import tensorflow as tf


# Static variables
FILES_ID = ["16mQThTqt55T4gyYcM1jFSazX1rrgZvRC","1UExXydQ_eES8PUOUY851J1bGpzSHNSOG", "1_zHnhIcm5NXxFV4kAqUDA6BHr97FoJdv"]
PATH_TO_STORE_MODEL = "./models/"
FILE_NAMES = ["YOLO_Face.h5", "yolo_anchors.txt", "face_classes.txt"]

def load_model():
    """Load pretrained facenet model
    """

    download_model()
    return _YOLO()

def download_model():
    """Download facenet h5 file from google drive
    """
    for i in range(len(FILE_NAMES)):
        if not Path(PATH_TO_STORE_MODEL + FILE_NAMES[i]).exists():
            print("Downloading",FILE_NAMES[i],"...")

            # Make directory to store downloaded model
            Path(PATH_TO_STORE_MODEL).mkdir(
                parents=True, exist_ok=True,
            )

            # Download from google drives
            download_file_from_google_drive(FILES_ID[i], PATH_TO_STORE_MODEL + FILE_NAMES[i])

##########################################################################################################################################
import os
import colorsys
import numpy as np
from tensorflow.compat.v1.keras import backend as K
from PIL import Image

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class _YOLO(object):
    def __init__(self,iou_threshold=0.45, score_threshold=0.5, model_path="models\YOLO_Face.h5", classes_path="models\\face_classes.txt", anchors_path="models\yolo_anchors.txt", img_size=(416, 416)):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.class_names = self._get_class(classes_path)
        self.anchors = self._get_anchors(anchors_path)
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self._generate(model_path)
        self.model_image_size = img_size

        pass
    
    def _get_class(self, classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def _get_anchors(self, anchors_path):
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _generate(self, model_path):
        model_path = os.path.expanduser(model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file'

        # load model, or construct model and load weights
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = tf.compat.v1.keras.models.load_model(model_path, compile=False)
        except:
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (
                           num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print(
            '*** {} model, anchors, and classes loaded.'.format(model_path))

        # generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = self._eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names),
                                           self.input_image_shape,
                                           score_threshold=self.score_threshold,
                                           iou_threshold=self.iou_threshold)
        return boxes, scores, classes

    def _eval(self,outputs, anchors, num_classes, image_shape,
            max_boxes=20, score_threshold=.6, iou_threshold=.5):
        '''Evaluate the YOLO model on given input and return filtered boxes'''

        num_layers = len(outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [
            [3, 4, 5], [1, 2, 3]]
        input_shape = K.shape(outputs[0])[1:3] * 32
        boxes = []
        box_scores = []

        for l in range(num_layers):
            _boxes, _box_scores = self._boxes_and_scores(outputs[l],
                                                anchors[anchor_mask[l]],
                                                num_classes, input_shape,
                                                image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)

        boxes = K.concatenate(boxes, axis=0)
        box_scores = K.concatenate(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []

        for c in range(num_classes):
            # TODO: use Keras backend instead of tf.
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor,
                iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        return boxes_, scores_, classes_

    def _boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        '''Process Convolutional layer output'''
        box_xy, box_wh, box_confidence, box_class_probs = self._yolo_head(feats,anchors,num_classes,input_shape)
        boxes = self._correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = K.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = K.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores
    
    def _yolo_head(self, feats, anchors, num_classes, input_shape, calc_loss=False):
        '''Convert final layer features to bounding box parameters'''

        num_anchors = len(anchors)
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        # height, width
        grid_shape = K.shape(feats)[1:3]
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                        [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                        [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1],
                                                            K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1],
                                                                K.dtype(feats))
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def _correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        '''Get corrected boxes'''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = K.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])

        # Scale boxes back to original image shape.
        boxes *= K.concatenate([image_shape, image_shape])
        return boxes
    
    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[
                       0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[
                       1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = self._letterbox_image(image, tuple(
                reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = self._letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        print(image_data.shape)
        image_data /= 255.
        # add batch dimension
        image_data = np.expand_dims(image_data, 0)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
       
        return out_scores, out_boxes, out_classes

    def _letterbox_image(self,image, size):
        '''Resize image with unchanged aspect ratio using padding'''

        img_width, img_height = image.size
        w, h = size
        scale = min(w / img_width, h / img_height)
        nw = int(img_width * scale)
        nh = int(img_height * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def close_session(self):
        self.sess.close()



