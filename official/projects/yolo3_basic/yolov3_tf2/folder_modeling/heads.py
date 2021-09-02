
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

from modeling.darknet import ConvModule, BlockModule, Darknet  #projects.yolo3.

"""
For each x_36, x_61 skip connection FCN
"""
class GenModule(tf.keras.layers.Layer):
    def __init__(self, filters, name="darkent53", **kwargs):
        super(GenModule, self).__init__()
        # with skip connection
        # self.skip_input = Input()
        self.conv_0 = ConvModule(filters, 1)
        self.upsample_0 = UpSampling2D(2)
        self.concate_0 = Concatenate()
        # 
        self.conv_1 = ConvModule(filters, 1)
        self.conv_2 = ConvModule(filters * 2, 3)

    def call(self, x):
        if isinstance(x, tuple):
            # inputs = Input(x[0].shape[1:]), Input(x[1].shape[1:])
            x, x_skip = x
            # x, x_skip = tf.expand_dims(x[0], axis=0), tf.expand_dims(x[1], axis=0)
            # concat with skip connection
            x = self.conv_0(x)
            x = self.upsample_0(x)
            x = self.concate_0([x, x_skip])
        # else:
            # x = inputs = Input(x.shape[1:])
            # x = tf.expand_dims(x, axis=0)

        # x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_1(x)
        # x = self.conv_2(x)
        # x = self.conv_1(x)

        return x


class PredModule(tf.keras.layers.Layer):
    def __init__(self, filters, num_anchors=3, num_classes=80, name=None, **kwargs):
        super(PredModule, self).__init__()
        # self.x_input = Input()
        self.conv_0 = ConvModule(filters * 2, 3)
        self.conv_1 = ConvModule(num_anchors * (num_classes + 5), 1, batch_norm=False)
        self.reshape_1 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            num_anchors, num_classes + 5)))

    def call(self, x_in):
        # x = inputs = self.x_input(x_in.shape[1:])
        x = inputs = x_in
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.reshape_1(x)
        return x


class PredModel(tf.keras.Model):
    def __init__(self, filters, name=None, **kwargs):
        super(PredModel, self).__init__()
        self.gen_conv = GenModule(filters, name='yolo_conv')
        self.pred_output = PredModule(filters, name='yolo_output')

    def call(self, x_in):
        x = x_in 
        x = self.gen_conv(x)
        x = self.pred_output(x)
        return x


# Tests
if __name__ == '__main__':

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
    print('Test len of outputs: ', len(y))
    x_36, x_61, x = y

    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # print('\nTest custom model: GenModule() w/o skip connections')
    # gen_m = GenModule(512)
    # x = gen_m(x)
    # print('Test weights:', len(gen_m.weights))
    # print('Test trainable weights:', len(gen_m.trainable_weights))

    # print('\nTest custom model: PredModule()')
    # pred_m = PredModule(512, len(masks[0]))
    # y = pred_m(x)
    # print('Test weights:', len(pred_m.weights))
    # print('Test trainable weights:', len(pred_m.trainable_weights))

    # print('\nTest custom model: GenModule() w/ skip connections x_61')
    # gen_m = GenModule(256)
    # print(x.shape, x_61.shape)
    # x = gen_m((x, x_61))
    # print('Test weights:', len(gen_m.weights))
    # print('Test trainable weights:', len(gen_m.trainable_weights))

    # print('\nTest custom model: PredModule()')
    # pred_m = PredModule(256, len(masks[1]))
    # y = pred_m(x)
    # print('Test weights:', len(pred_m.weights))
    # print('Test trainable weights:', len(pred_m.trainable_weights))

    # print('\nTest custom model: GenModule() w/ skip connections x_36')
    # gen_m = GenModule(128)
    # print(x.shape, x_36.shape)
    # x = gen_m((x, x_36))
    # print('Test weights:', len(gen_m.weights))
    # print('Test trainable weights:', len(gen_m.trainable_weights))

    # print('\nTest custom model: PredModule()')
    # pred_m = PredModule(128, len(masks[1]))
    # y = pred_m(x)
    # print('Test weights:', len(pred_m.weights))
    # print('Test trainable weights:', len(pred_m.trainable_weights))

    print('\n---')
    head_m = PredModel(512)
    y = head_m(x)
    print('Test weights:', len(head_m.weights))
    print('Test trainable weights:', len(head_m.trainable_weights))    