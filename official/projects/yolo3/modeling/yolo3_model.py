
import tensorflow as tf
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
from tensorflow.python import training 
from projects.yolo3.modeling.darknet import DarknetConv, DarknetBlock, Darknet
from projects.yolo3.modeling.heads import YoloConv, YoloOutput
from projects.yolo3.modeling.bbox import BBox, NMS




DARKNET53 = {
    'input_spec': [None, None, 3], 
    'backbone': [
        [DarknetConv(),    32, 3, False], # Conv(),  filters, size,    isOutput
        [DarknetBlock(),   64, 1, False], # Block(), filters, repeats, isOutput
        [DarknetBlock(),  128, 2, False], 
        [DarknetBlock(),  256, 8, True ], 
        [DarknetBlock(),  512, 8, True ], 
        [DarknetBlock(), 1024, 4, True ]
    ], 
    'head': [
        [YoloConv(),   512, None, False], # filters, hasSkipConn, isOutput
        [YoloOutput(), 512, None, True ], 
        [YoloConv(),   256, '61', False], 
        [YoloOutput(), 256, None, True ], 
        [YoloConv(),   128, '36', False], 
        [YoloOutput(), 128, None, True ], 
    ],
    'masks': [], 
    'anchors': [], 
}



"""
YOLOv3 Model
"""

class YOLOv3(tf.keras.Model): 
    def __init__(self, cfg_model=DARKNET53, training=False, name="yolo3", **kwargs): 
        super(YOLOv3).__init__(name=None, **kwargs)
        self.input = Input(cfg_model['input_spec'], name='input')
        self.anchors = cfg_model['anchors']
        self.masks = cfg_model['masks']
        self.backbone = cfg_model['backbone']
        self.head = cfg_model['head']

    def call(self, x_in):
        outputs = []
        x = inputs = self.input(x_in)
        x_36, x_61, x = Darknet(backbone=self.backbone, name="darknet53")(x)
        # head
        for (head_fn, filters, skip, isOutput) in self.head: 
            if skip == '61': 
                x = head_fn(filters)((x, x_61))
            elif skip == '36':
                x = head_fn(filters)((x, x_36))
            else: 
                x = head_fn(filters)(x)
            if isOutput: outputs.append(x)
        # ready for bbox
        output_0, output_1, output_2 = outputs 
        
        # TODO: if training: Model(inputs, outputs, name='yolo3_train')

        boxes_0 = BBox(self.anchors[self.masks[0]])(output_0)
        boxes_1 = BBox(self.anchors[self.masks[1]])(output_1)
        boxes_2 = BBox(self.anchors[self.masks[2]])(output_2)

        # anchors, masks, max_boxes, iou_threshold, score_threshold
        outputs = NMS(self.anchors, self.masks)((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
