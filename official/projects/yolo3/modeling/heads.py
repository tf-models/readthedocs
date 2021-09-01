
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

from projects.yolo3.modeling.darknet import *

"""
For each x_36, x_61 skip connection FCN
"""
class YoloConv(tf.keras.Model):
    def __init__(self, filters, name="darkent53", **kwargs):
        super(YoloConv).__init__(name=name, **kwargs)
        self.filters = filters 

    def call(self, x):
        if isinstance(x, tuple):
            inputs = Input(x[0].shape[1:]), Input(x[1].shape[1:])
            x, x_skip = inputs
            # concat with skip connection
            x = DarknetConv(self.filters, 1)(x)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x.shape[1:])

        x = DarknetConv(self.filters,     1)(x)
        x = DarknetConv(self.filters * 2, 3)(x)
        x = DarknetConv(self.filters,     1)(x)
        x = DarknetConv(self.filters * 2, 3)(x)
        x = DarknetConv(self.filters,     1)(x)
        return x


class YoloOutput(tf.keras.Model):
    def __init__(self, filters, num_anchors=3, num_classes=80, name=None, **kwargs):
        super(YoloOutput).__init__(name=name, **kwargs)
        self.filters = filters 
        self.num_anchors = num_anchors 
        self.num_classes = num_classes

    def call(self, x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, self.filters * 2, 3)
        x = DarknetConv(x, self.num_anchors * (self.num_classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            self.num_anchors, self.num_classes + 5)))(x)
        return x

