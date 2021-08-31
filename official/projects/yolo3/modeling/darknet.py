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


class Darknet(tf.keras.Model):

    def __init__(
        self, 
        anchors=3, 
        num_classes=80):
        # super(Darknet).__init__()
        
        self.input = None 
        self.config = [] 
        self.anchors=anchors,
        self.num_classes=num_classes


    """
    basic conv used in classification and regression parts
    """
    def DarknetConv(self, x, filters, size, strides=1, batch_norm=True):
        if strides == 1:
            padding = 'same'
        else:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
            padding = 'valid'
        x = Conv2D(
                filters=filters, 
                kernel_size=size,
                strides=strides, padding=padding,
                kernel_regularizer=l2(0.0005))(x)
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x


    """
    Darknet53 architecture for class probabilities 
    """
    def DarknetResidual(self, x, filters):
        prev = x
        x = self.DarknetConv(x, filters // 2, 1)
        x = self.DarknetConv(x, filters, 3)
        x = Add()([prev, x])
        return x


    def DarknetBlock(self, x, filters, blocks):
        x = self.DarknetConv(x, filters, 3, strides=2)
        for _ in range(blocks):
            x = self.DarknetResidual(x, filters)
        return x


    def Darknet53(self, name=None):
        x = inputs = Input([None, None, 3])
        x = self.DarknetConv(x, 32, 3)
        x = self.DarknetBlock(x, 64, 1)
        x = self.DarknetBlock(x, 128, 2)  # skip connection
        x = x_36 = self.DarknetBlock(x, 256, 8)  # skip connection
        x = x_61 = self.DarknetBlock(x, 512, 8)
        x = self.DarknetBlock(x, 1024, 4)
        return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


    """
    3 Outputs to predicts bounding boxes
    """
    def YoloConv(self, x_in, filters, name=None):
        # 3 output for x_36, x_61, x_output
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs
            # concat with skip connection
            x = self.DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = self.DarknetConv(x, filters, 1)
        x = self.DarknetConv(x, filters * 2, 3)
        x = self.DarknetConv(x, filters, 1)
        x = self.DarknetConv(x, filters * 2, 3)
        x = self.DarknetConv(x, filters, 1)

        return tf.keras.Model(inputs, x, name=name)(x_in)


    def YoloOutput(self, x_in, filters, anchors, num_classes, name=None):
        x = inputs = Input(x_in.shape[1:])
        x = self.DarknetConv(x, filters * 2, 3)
        x = self.DarknetConv(x, anchors * (num_classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, num_classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)


