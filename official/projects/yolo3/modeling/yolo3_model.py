


# from absl import flags
# from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.losses import (
#     binary_crossentropy,
#     sparse_categorical_crossentropy
# )
# from .utils import broadcast_iou

# from ops import broadcast_iou

from bbox import *
from darknet import * 
from projects.yolo3.configs import * as cfg

## config 
# yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), 
#                          (30, 61), (62, 45), (59, 119), 
#                          (116, 90), (156, 198), (373, 326)],
#                         np.float32) / 416

# yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


## modeling component
class Yolo3Model(tf.keras.Model):

    def __init__(
        self, 
        num_classes=80, 
        input_size=None, 
        input_channels=3,
        anchors=yolo_anchors, 
        masks=yolo_anchor_masks, 
        ):

        self.InputSpec=Input([input_size, input_size, input_channels], name='input')

        self.Darknet=Darknet(
            masks=masks, 
            num_classes=num_classes,
        )

        self.BBox=BBox(
            anchors=anchors, 
            masks=masks, 
            max_boxes=100, 
            iou_threshold=0.5, 
            score_threshold=0.5, 
            num_classes=num_classes, 
        )
        # self.anchors=yolo_anchors
        # self.masks=yolo_anchor_masks

    def call(self, input):

        x = inputs = self.InputSpec()(input)

        x_36, x_61, x = self.Darknet.Darknet53(name='yolo_darknet')(x)

        x = self.Darknet.YoloConv(filters=512, name='yolo_conv_0')(x)
        output_0 = self.Darknet.YoloOutput(filters=512, name='yolo_output_0')(x)

        x = self.Darknet.YoloConv(filters=256, name='yolo_conv_1')((x, x_61))
        output_1 = self.Darknet.YoloOutput(filters=256, name='yolo_output_1')(x)

        x = self.Darknet.YoloConv(filters=128, name='yolo_conv_2')((x, x_36))
        output_2 = self.Darknet.YoloOutput(filters=128, name='yolo_output_2')(x)

        # if training:
        #     return Model(inputs, (output_0, output_1, output_2), name='yolov3')

        # bbox, objectness, class_probs, pred_box
        boxes_0 = self.BBox.YoloBoxes(pred=x, anchors_mask_i=0, name='yolo_boxes_0')(output_0)
        boxes_1 = self.BBox.YoloBoxes(pred=x, anchors_mask_i=1, name='yolo_boxes_1')(output_1)
        boxes_2 = self.BBox.YoloBoxes(pred=x, anchors_mask_i=2, name='yolo_boxes_2')(output_2)

        outputs = self.BBox.YoloNMS(output=x, name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return Model(inputs, outputs, name='yolov3')
