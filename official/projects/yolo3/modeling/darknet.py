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
from tensorflow.keras.regularizers import l2

"""
Layers for Darknet-53 Architecture:
    1. DarknetConv (basic conv)
    2. DarknetResidual (build up with conv and add layers)
    3. DarknetBlock (repeat residual block)
"""

class DarknetConv(tf.keras.layers.Layer):
    """ 1 Basic Conv"""
    def __init__(self, filters, size, strides=1, batch_norm=True, name="darknet_conv", **kwargs):
        super(DarknetConv, self).__init__(name=name, **kwargs)
        self.filters=filters
        self.size=size
        self.strides=strides
        self.batch_norm=batch_norm

    def call(self, x):
        if self.strides == 1:
            padding = 'same'
        else:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
            padding = 'valid'
        x = Conv2D(
                filters=self.filters, 
                kernel_size=self.size,
                strides=self.strides, padding=padding,
                kernel_regularizer=l2(0.0005))(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x

class DarknetResidual(tf.keras.layers.Layer):
    """ 2 DarknetConv + 1 AddLayer"""
    def __init__(self, filters, name="darknet_residual", **kwargs):
        super(DarknetResidual).__init__(name=name, **kwargs)
        # self.filters=filters 
        self.conv_0 = DarknetConv(filters//2, 1)
        self.conv_1 = DarknetConv(filters, 3)
    
    def call(self, x):
        prev = x 
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = Add()([prev, x])
        return x 

class DarknetBlock(tf.keras.layers.Layer):
    """ 1 DarknetConv + Repeat DarknetResidual blocks"""
    def __init__(self, filters, blocks, name="darknet_block", **kwargs):
        super(DarknetBlock).__init__(name=name, **kwargs)
        self.conv = DarknetConv(filters, 3, strides=2)
        self.resblock = DarknetResidual(filters)
        self.blocks = blocks

    def call(self, x):
        x = self.conv(x)
        for _ in range(self.blocks):
            x = self.resblock(x)
        return x


"""
Models for YOLO3: 
    1. Classification: architecture darknet53
    2. Location: 3 outputs 
"""

class Darknet(tf.keras.Model):
    def __init__(self, backbone, name="darkent53", **kwargs):
        super(Darknet).__init__(name=name, **kwargs)
        # 3 type layers
        # self.input = Input([in_size, in_size, in_channels], name='input')
        self.backbone = backbone

    def call(self, x):
        outputs = []
        # x = inputs = self.input(x)
        for (fn, filters, para, isOutput) in self.backbone:
            x = fn(filters, para)(x)
            if isOutput: outputs.append(x)
        x_36, x_61, x = outputs
        return x_36, x_61, x

