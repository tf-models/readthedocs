import tensorflow as tf
# from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add, Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization # Concatenate, Lambda, MaxPool2D, UpSampling2D,
)
from tensorflow.keras.regularizers import l2

"""
Layers for Darknet-53 Architecture:
    1. DarknetConv (basic conv)
    2. DarknetResidual (build up with conv and add layers)
    3. DarknetBlock (repeat residual block)
"""

class ConvModule(tf.keras.layers.Layer):
    """ 1 Basic Conv """
    def __init__(self, filters, size, strides=1, batch_norm=True, name="darknet_conv", **kwargs):
        super(ConvModule, self).__init__()
        self.strides = strides
        if strides == 1:
            padding = 'same'
        else:
            self.zero_padding = ZeroPadding2D(((1, 0), (1, 0))) # top left half-padding
            padding = 'valid'
        self.conv2d = Conv2D(
                filters=filters, 
                kernel_size=size,
                strides=strides, padding=padding,
                kernel_regularizer=l2(0.0005))

        self.batch_norm = batch_norm
        self.bn = BatchNormalization()
        self.leaky_relu = LeakyReLU(alpha=0.1)

    def call(self, x_in, training=False):
        if self.strides != 1:
            x_in = self.zero_padding(x_in)  # top left half-padding
        x = self.conv2d(x_in)
        if self.batch_norm:
            x = self.bn(x, training=training)
            x = self.leaky_relu(x)
        return x


class ResModule(tf.keras.layers.Layer):
    """ 2 DarknetConv + 1 AddLayer """
    def __init__(self, filters, name="darknet_residual", **kwargs):
        super(ResModule, self).__init__()
        self.conv_0 = ConvModule(filters//2, 1)
        self.conv_1 = ConvModule(filters, 3)
        self.add = Add()
    
    def call(self, x):
        prev = x 
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.add([prev, x])
        return x 


class BlockModule(tf.keras.layers.Layer):
    """ 1 DarknetConv + Repeat DarknetResidual blocks """
    def __init__(self, filters, blocks, name="darknet_block", **kwargs):
        super(BlockModule, self).__init__()
        self.conv = ConvModule(filters, 3, strides=2)
        self.resblock = ResModule(filters)
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
    def __init__(self, backbone, name="darkent53", **kwargs):
        super(Darknet, self).__init__()
        self.modules = []
        for (fn, filters, para, isOutput) in backbone:
            self.modules.append( (fn(filters, para), isOutput) )

    def call(self, input_tensor, training=False, **kwargs):
        outputs = []
        x = input_tensor
        for (module_layer, isOutput) in self.modules:
            x = module_layer(x)
            if isOutput: outputs.append(x)
        x_36, x_61, x = outputs
        return x_36, x_61, x

    def build_graph(self, raw_shape):
        x = Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


# Tests
if __name__ == '__main__':

    print('\nTest custom layer: ConvModule()')
    conv_m = ConvModule(32, 3)
    y = conv_m(tf.ones(shape=(2, 32, 32, 3)))
    print('Test weights:', len(conv_m.weights))
    print('Test trainable weights:', len(conv_m.trainable_weights))

    print('\nTest custom layer: ResModule()')
    res_m = ResModule(64, 1)
    y = res_m(tf.ones(shape=(2, 32, 32, 64)))
    print('Test weights:', len(res_m.weights))
    print('Test trainable weights:', len(res_m.trainable_weights))

    print('\nTest custom layer: BlockModule()')
    block_m = BlockModule(128, 2)
    y = block_m(tf.ones(shape=(2, 32, 32, 64)))
    print('Test weights:', len(block_m.weights))
    print('Test trainable weights:', len(block_m.trainable_weights))

    print('\nTest custom model: Darknet()')
    backbone = [
        [ConvModule,    32, 3, False], # fn_conv(),  filters,    size, isOutput
        [BlockModule,   64, 1, False], # fn_block(), filters, repeats, isOutput
        [BlockModule,  128, 2, False], 
        [BlockModule,  256, 8, True ], 
        [BlockModule,  512, 8, True ], 
        [BlockModule, 1024, 4, True ]
    ]
    raw_input = (32, 32, 3)
    darkn = Darknet(backbone=backbone)
    y = darkn(tf.ones(shape=(0, *raw_input)))
    darkn.build_graph(raw_input).summary()
