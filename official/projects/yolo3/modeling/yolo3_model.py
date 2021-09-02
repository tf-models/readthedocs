

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input,
    InputSpec, 
    Lambda,
)
from tensorflow.python import training 
from modeling.darknet import ConvModule, BlockModule, Darknet # projects.yolo3.modeling.
from modeling.heads import GenModule, PredModule, PredModel
from modeling.bbox import yolo_boxes, yolo_nms #BBox, NMS




DARKNET53 = {
    'input_spec': [None, None, 3], 
    'backbone': [
        [ConvModule,    32, 3, False], # fn_conv(),  filters,    size, isOutput
        [BlockModule,   64, 1, False], # fn_block(), filters, repeats, isOutput
        [BlockModule,  128, 2, False], 
        [BlockModule,  256, 8, True ], 
        [BlockModule,  512, 8, True ], 
        [BlockModule, 1024, 4, True ]
    ], 
    # 'head': [
    #     [GenModule,  512, 'yolo_conv_0',   None, False], # fn(), filters, name, hasSkipConn, isOutput
    #     [PredModule, 512, 'yolo_output_0', None, True ], 
    #     [GenModule,  256, 'yolo_conv_1',   '61', False], 
    #     [PredModule, 256, 'yolo_output_1', None, True ], 
    #     [GenModule,  128, 'yolo_conv_2',   '36', False], 
    #     [PredModule, 128, 'yolo_output_2', None, True ], 
    # ],
    'masks': np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]), 
    'anchors': np.array([(10, 13),  (16, 30),   (33, 23), 
                         (30, 61),  (62, 45),   (59, 119), 
                         (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416.0 ,
    'classes': 80
}



"""
YOLOv3 Model
"""

class YOLOv3(tf.keras.Model): 
    def __init__(self, cfg_model=DARKNET53, name='yolo3', 
        input_specs=InputSpec(shape=[None, None, None, 3]), **kwargs): 

        super(YOLOv3, self).__init__()
        # self.input_spec = Input([None, None, 3], name='input') #cfg_model['input_spec']

        self.darknet = Darknet(backbone=cfg_model['backbone'], name='yolo_darknet')
        
        self.output_0 = PredModel(512)
        self.output_1 = PredModel(256)
        self.output_2 = PredModel(128)

        anchors = cfg_model['anchors']
        masks = cfg_model['masks']
        classes = cfg_model['classes']

        self.bbox_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')
        self.bbox_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')
        self.bbox_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')
        self.nms = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name='yolo_nms')

    def call(self, x_in):
        outputs = []
        # x = inputs = self.input_spec(x_in)
        x = x_in
        x = inputs = Input([None, None, 3], name='input')
        x_36, x_61, x = self.darknet(x)
        # head
        output_0 = self.output_0(x)
        output_1 = self.output_1((x, x_61))
        output_2 = self.output_2((x, x_36))
        # ready for bbox
        
        # TODO: if training: Model(inputs, outputs, name='yolo3_train')

        boxes_0 = self.bbox_0(output_0)
        boxes_1 = self.bbox_1(output_1)
        boxes_2 = self.bbox_2(output_2)

        # anchors, masks, max_boxes, iou_threshold, score_threshold
        outputs = self.nms((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return outputs



# Tests
if __name__ == '__main__':

    print('\nTest custom layer: YOLOv3()')
    yolo_m = YOLOv3(cfg_model=DARKNET53)
    y = yolo_m(tf.ones(shape=(2, 32, 32, 3)))
    print('Test weights:', len(yolo_m.weights))
    print('Test trainable weights:', len(yolo_m.trainable_weights))