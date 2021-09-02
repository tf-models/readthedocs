

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    InputSpec, 
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.python import training 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

# from modeling.darknet import ConvModule, BlockModule, Darknet # projects.yolo3.modeling.
# from modeling.heads import GenModule, PredModule, PredModel
# from modeling.bbox import yolo_boxes, yolo_nms #BBox, NMS






"""
Layers for Darknet-53 Architecture:
    1. DarknetConv (basic conv)
    2. DarknetResidual (build up with conv and add layers)
    3. DarknetBlock (repeat residual block)
"""

class DarknetCov(tf.keras.layers.Layer):
    """ 1 Basic Conv """
    def __init__(self, filters, size, strides=1, batch_norm=True, name="darknet_conv", **kwargs):
        super(DarknetCov, self).__init__(name=name, **kwargs)
        self.strides = strides
        if strides == 1:
            padding = 'same'
        else:
            self.zero_padding = ZeroPadding2D(((1, 0), (1, 0))) # top left half-padding
            padding = 'valid'
        self.conv2d = Conv2D(filters, size, strides, padding, kernel_regularizer=l2(0.0005))

        self.batch_norm = batch_norm
        self.bn = BatchNormalization()
        self.leaky_relu = LeakyReLU(alpha=0.1)

    def call(self, x):
        if self.strides != 1:
            x = self.zero_padding(x)  # top left half-padding
        x = self.conv2d(x)
        if self.batch_norm:
            x = self.bn(x)
            x = self.leaky_relu(x)
        return x



class DarknetRes(tf.keras.layers.Layer):
    """ 2 DarknetConv + 1 AddLayer """
    def __init__(self, filters, name="darknet_residual", **kwargs):
        super(DarknetRes, self).__init__(name=name, **kwargs)
        self.conv_0 = DarknetCov(filters//2, 1)
        self.conv_1 = DarknetCov(filters, 3)
        self.add = Add()
    
    def call(self, x):
        prev = x 
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.add([prev, x])
        return x 



class DarknetBlock(tf.keras.layers.Layer):
    """ 1 DarknetConv + Repeat DarknetResidual blocks """
    def __init__(self, filters, blocks, name="darknet_block", **kwargs):
        super(DarknetBlock, self).__init__()
        self.conv = DarknetCov(filters, 3, strides=2)
        self.resblock = DarknetRes(filters)
        self.blocks = blocks

    def call(self, x):
        x = self.conv(x)
        for _ in range(self.blocks):
            x = self.resblock(x)
        return x

"""
Models for YOLO3 Architecture: 
    1. Classification: architecture darknet53
    2. Location: 3 outputs 
"""

class Darknet(tf.keras.Model):
    def __init__(self, backbone, name="yolo_darknet", **kwargs):
        super(Darknet, self).__init__(name=name, **kwargs)
        self.modules = []
        for (backbone_fn, filters, para, isOutput) in backbone:
            self.modules.append( (backbone_fn(filters, para), isOutput) )

    def call(self, x):
        outputs = []
        for (backbone_layer, isOutput) in self.modules:
            x = backbone_layer(x)
            if isOutput: 
                outputs.append(x)
        x_36, x_61, x = outputs
        return x_36, x_61, x

    def build_graph(self, x):
        x = Input(shape=x.shape[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))



"""
For each x_36, x_61 skip connection FCN
"""
class YoloConv(tf.keras.Model):
    def __init__(self, filters, name="yolo_conv", **kwargs):
        super(YoloConv, self).__init__(name=name, **kwargs)
        # with skip connection
        # self.conv_0 = DarknetCov(filters, 1)
        # self.upsample_0 = UpSampling2D(2)
        # self.concate_0 = Concatenate()
        # 
        self.conv_1 = DarknetCov(filters, 1)
        self.conv_2 = DarknetCov(filters*2, 3)
        self.conv_3 = DarknetCov(filters, 1)
        self.conv_4 = DarknetCov(filters*2, 3)
        self.conv_5 = DarknetCov(filters, 1)

    def call(self, x):

        x = self.conv_1(x)
        print('conv_1 1:', x.shape)
        x = self.conv_2(x)
        print('conv_2 1:', x.shape)
        x = self.conv_3(x)
        print('conv_1 2:', x.shape)
        x = self.conv_4(x)
        print('conv_2 2:', x.shape)
        x = self.conv_5(x)
        print('conv_1 3:', x.shape)
        return x

    def build_graph(self, x):
        x = Input(shape=x.shape[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class YoloSkip(tf.keras.Model):
    def __init__(self, filters, name="yolo_conv", **kwargs):
        super(YoloSkip, self).__init__(name=name, **kwargs)
        # with skip connection
        self.conv_0 = DarknetCov(filters, 1)
        self.upsample_0 = UpSampling2D(2)
        self.concate_0 = Concatenate(axis=-1)
        # 
        self.conv_1 = DarknetCov(filters, 1)
        self.conv_2 = DarknetCov(filters*2, 3)
        self.conv_3 = DarknetCov(filters, 1)
        self.conv_4 = DarknetCov(filters*2, 3)
        self.conv_5 = DarknetCov(filters, 1)

    def call(self, x_in):
        x, x_skip = x_in
        print(x_in[0].shape, x_in[1].shape)
        # concat with skip connection
        x = self.conv_0(x)
        x = self.upsample_0(x)
        print(x.shape, x_skip.shape)
        x = self.concate_0([x, x_skip])
        print(x.shape, x_skip.shape)

        x = self.conv_1(x)
        print('conv_1 1:', x.shape)
        x = self.conv_2(x)
        print('conv_2 1:', x.shape)
        x = self.conv_3(x)
        print('conv_1 2:', x.shape)
        x = self.conv_4(x)
        print('conv_2 2:', x.shape)
        x = self.conv_5(x)
        print('conv_1 3:', x.shape)
        return x

    def build_graph(self, x):
        x = Input(shape=x[0].shape[1:]), Input(shape=x[1].shape[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))



class PredConv(tf.keras.Model):
    def __init__(self, filters, num_anchors=3, num_classes=80, name='yolo_output', **kwargs):
        super(PredConv, self).__init__(name=name, **kwargs)
        self.conv_0 = DarknetCov(filters * 2, 3)
        self.conv_1 = DarknetCov(num_anchors * (num_classes + 5), 1, batch_norm=False)
        self.reshape_1 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],num_anchors, num_classes + 5)))

    def call(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.reshape_1(x)
        return x

    def build_graph(self, x):
        x = Input(shape=x.shape[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))



def _meshgrid(n_a, n_b):
    
    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = _meshgrid(grid_size[1],grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes, max_boxes=100):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs

    dscores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(dscores,[1])
    bbox = tf.reshape(bbox,(-1,4))
    classes = tf.argmax(dscores,1)

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=100,
        iou_threshold=0.5,
        score_threshold=0.5,
        soft_nms_sigma=0.5
    )
    
    num_valid_nms_boxes = tf.shape(selected_indices)[0]
    substract_valid = max_boxes - num_valid_nms_boxes
    selected_indices = tf.concat([selected_indices,tf.zeros(substract_valid, tf.int32)], 0)
    selected_scores = tf.concat([selected_scores,tf.zeros(substract_valid, tf.float32)], -1)

    boxes = tf.gather(bbox, selected_indices)
    boxes = tf.expand_dims(boxes, axis=0)
    scores=selected_scores
    scores = tf.expand_dims(scores, axis=0)
    classes = tf.gather(classes,selected_indices)
    classes = tf.expand_dims(classes, axis=0)
    valid_detections=num_valid_nms_boxes
    valid_detections = tf.expand_dims(valid_detections, axis=0)

    return boxes, scores, classes, valid_detections


"""
YOLOv3 Model
"""
DARKNET53 = {
    'input_spec': [None, None, 3], 
    'backbone': [
        [DarknetCov,     32, 3, False], # fn_conv(),  filters,    size, isOutput
        [DarknetBlock,   64, 1, False], # fn_block(), filters, repeats, isOutput
        [DarknetBlock,  128, 2, False], 
        [DarknetBlock,  256, 8, True ], 
        [DarknetBlock,  512, 8, True ], 
        [DarknetBlock, 1024, 4, True ]
    ], 
    'masks': np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]), 
    'anchors': np.array([(10, 13),  (16, 30),   (33, 23), 
                         (30, 61),  (62, 45),   (59, 119), 
                         (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416.0 ,
    'classes': 80
}


class YOLOv3(tf.keras.Model): 
    def __init__(self, cfg_model=DARKNET53, name='yolo3', 
        input_specs=InputSpec(shape=[None, None, None, 3]), **kwargs): 

        super(YOLOv3, self).__init__()
        # self.input_spec = Input([None, None, 3], name='input') #cfg_model['input_spec']

        self.darknet = Darknet(backbone=cfg_model['backbone'], name='yolo_darknet')
        anchors, masks, classes = cfg_model['anchors'], cfg_model['masks'], cfg_model['classes']

        self.y_conv_0 = YoloConv(512, name='yolo_conv_0')
        self.output_0 = PredConv(512, len(masks[0]), name='yolo_output_0')

        self.y_conv_1 = YoloSkip(256, name='yolo_conv_1')
        self.output_1 = PredConv(256, len(masks[1]), name='yolo_output_1')

        self.y_conv_2 = YoloSkip(128, name='yolo_conv_2')
        self.output_2 = PredConv(128, len(masks[2]), name='yolo_output_2')

        self.bbox_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')
        self.bbox_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')
        self.bbox_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')
        self.nms = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name='yolo_nms')

    def call(self, x):
        outputs = []
        # x = x_in
        # x = inputs = Input([None, None, 3], name='input')
        x_36, x_61, x = self.darknet(x)
        # head
        x        = self.y_conv_0(x)
        output_0 = self.output_0(x)
        x        = self.y_conv_1((x, x_61))
        output_1 = self.output_1(x)
        x        = self.y_conv_2((x, x_36))
        output_2 = self.output_2(x)
        # ready for bbox
        
        # TODO: if training: Model(inputs, outputs, name='yolo3_train')

        boxes_0 = self.bbox_0(output_0)
        boxes_1 = self.bbox_1(output_1)
        boxes_2 = self.bbox_2(output_2)

        # anchors, masks, max_boxes, iou_threshold, score_threshold
        outputs = self.nms((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return outputs

    # def build(self, x):
    #     x = Input(shape=x.shape[1:])
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def build_graph(self, raw_input):
        x = Input(shape=raw_input)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

# Tests
if __name__ == '__main__':

    print('\nTest custom layer: YOLOv3()')
    yolo_m = YOLOv3(cfg_model=DARKNET53)
    y = yolo_m(tf.ones(shape=(2, 32, 32, 3)))
    print('Test weights:', len(yolo_m.weights))
    print('Test trainable weights:', len(yolo_m.trainable_weights))