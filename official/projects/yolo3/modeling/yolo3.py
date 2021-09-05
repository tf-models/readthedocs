
import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate, Conv2D, LeakyReLU, UpSampling2D, ZeroPadding2D, BatchNormalization 
)

"""
Part 1: Feature Extraction
"""
class BaseConv(tf.keras.Model):
    """ base conv includes padding, batchnorm, leakyrelu 
    Args: 
        strides: padding same if strides==1, otherwise padding valid. 
    """
    
    def __init__(self, filters, kernel_size, strides=1):
        super().__init__()
        # padding
        self.strides = strides
        self.padd = ZeroPadding2D(((1, 0), (1, 0)))
        self.basic0 = Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=('same'if strides == 1 else'valid'), 
            use_bias=False)
        # base conv
        self.basic1 = BatchNormalization()
        self.basic2 = LeakyReLU(alpha=0.1)

    def call(self, inputs, training=False):
        if self.strides > 1 :
            x = self.padd(inputs)
        else:
            x = inputs
        x = self.basic0(x)
        x = self.basic1(x, training=training)
        x = self.basic2(x)

        return x


class ResBlock(tf.keras.Model):
    """ residual block includes 2 base conv 
    Args:
        filters: first one base conv filters
        strides: same strides for both two base conv
    Return:
        input add into output
    """
    def __init__(self, filters, strides=1):
        super().__init__()
        self.basic0 = BaseConv(filters=filters, kernel_size=1, strides=strides)
        self.basic1 = BaseConv(filters=filters * 2, kernel_size=3, strides=strides)

    def call(self, inputs, training=False):
        x = self.basic0(inputs, training=training)
        x = self.basic1(x, training=training)
        x = x + inputs

        return x


class DarkNet53(tf.keras.Model):
    """ architecture to extract features 
    
    """
    def __init__(self):
        super().__init__()
        self.basic0 = BaseConv(filters=32, kernel_size=3)

        self.basic1 = BaseConv(filters=64, kernel_size=3, strides=2)
        self.multi0 = ResBlock(32)

        self.basic2 = BaseConv(filters=128, kernel_size=3, strides=2)
        self.multi1 = tf.keras.Sequential()
        for _ in range(2):
            self.multi1.add(ResBlock(64))

        self.basic3 = BaseConv(filters=256, kernel_size=3, strides=2)
        self.multi2 = tf.keras.Sequential()
        for _ in range(8):
            self.multi2.add(ResBlock(128))

        self.basic4 = BaseConv(filters=512, kernel_size=3, strides=2)
        self.multi3 = tf.keras.Sequential()
        for _ in range(8):
            self.multi3.add(ResBlock(256))

        self.basic5 = BaseConv(filters=1024, kernel_size=3, strides=2)
        self.multi4 = tf.keras.Sequential()
        for _ in range(4):
            self.multi4.add(ResBlock(512))

    def call(self, inputs, training=False):
        x = self.basic0(inputs, training=training)
        x = self.basic1(x, training=training)
        x = self.multi0(x, training=training)
        x = self.basic2(x, training=training)
        x = self.multi1(x, training=training)
        x = self.basic3(x, training=training)
        route0 = self.multi2(x, training=training)
        x = self.basic4(route0, training=training)
        route1 = self.multi3(x, training=training)
        x = self.basic5(route1, training=training)
        output = self.multi4(x, training)

        return route0, route1, output

"""
Part 2: Prediction
"""

class SkipConn(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.basic = BaseConv(filters=filters, kernel_size=1)
        self.upsample = UpSampling2D(2)
        self.concate = Concatenate()

    def call(self, inputs, training=False):
        route, route_i = inputs
        x = self.basic(route, training=training)
        x = self.upsample(x)
        x = self.concate([x, route_i])
        return x

class YoloConvBlock(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.basic0 = BaseConv(filters=filters, kernel_size=1)
        self.basic1 = BaseConv(filters=filters * 2, kernel_size=3)
        self.basic2 = BaseConv(filters=filters, kernel_size=1)
        self.basic3 = BaseConv(filters=filters * 2, kernel_size=3)
        self.basic4 = BaseConv(filters=filters, kernel_size=1)
        self.basic5 = BaseConv(filters=filters * 2, kernel_size=3)

    def call(self, inputs, training=False):
        x = self.basic0(inputs, training)
        x = self.basic1(x, training)
        x = self.basic2(x, training)
        x = self.basic3(x, training)
        route = self.basic4(x, training)
        output = self.basic5(route, training)
        return route, output


class DetectLayer(tf.keras.Model):
    def __init__(self, n_classes, anchors):
        super().__init__()
        self.conv = Conv2D(
            filters=len(anchors) * (n_classes + 5), 
            kernel_size=1, strides=1)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        return x


"""
YOLOv3 Model
"""
class YOLOV3(tf.keras.Model):

    def __init__(self, n_classes, anchors):
        super().__init__()
        self.n_classes = n_classes
        self.feature_extractor = DarkNet53()

        self.conv_block0 = YoloConvBlock(filters=512)
        self.conv_block1 = YoloConvBlock(filters=256)
        self.conv_block2 = YoloConvBlock(filters=128)
        
        self.skip_conn1 = SkipConn(filters=256)
        self.skip_conn2 = SkipConn(filters=128)

        self.detector0 = DetectLayer(n_classes, anchors=anchors[2])
        self.detector1 = DetectLayer(n_classes, anchors=anchors[1])
        self.detector2 = DetectLayer(n_classes, anchors=anchors[0])

    def call(self, inputs, training=False, finetuning=True):
        if finetuning:
            self.feature_extractor.trainable = False
        else:
            self.feature_extractor.trainable = True

        route0, route1, x = self.feature_extractor(inputs, training=False)

        route, x = self.conv_block0(x, training=training)
        detect0 = self.detector0(x, training=training)

        x = self.skip_conn1((route, route1))
        route, x = self.conv_block1(x, training=training)
        detect1 = self.detector1(x, training=training)

        x = self.skip_conn2((route, route0))
        _, x = self.conv_block2(x, training=training)
        detect2 = self.detector2(x, training=training)

        return detect0, detect1, detect2



if __name__ == '__main__':

    _ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
    model = YOLOV3(n_classes=1, anchors=_ANCHORS)
    model.build(input_shape=(None, 608, 608, 3))

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    # tf.config.threading.set_inter_op_parallelism_threads(8)
    # tf.config.threading.set_intra_op_parallelism_threads(8)
    for i, var in enumerate(model.variables):
        print(i, var.name)
        # names = var.name.split('/')
        # block, bias = names[0], names[-1]
        # if block.startswith('detect_layer') and bias.startswith('bias'):
        #     print(i, names)

    # test_set = make_dataset(BATCH_SIZE=8, file_name='test_tf_record', split=False)

    # for i in test_set:
    #     images = i[0]
    #     image_size = tf.cast(tf.shape(images)[1:3], tf.float32)
    #     label = i[1:]
    #     images = tf.cast(images, tf.float32) / 255.0
    #     detect0, detect1, detect2 = model(images, training=True, finetuning=True)
    #     de_de0 = decode(detect0, _ANCHORS[6:9], 1, image_size)
    #     total_loss = yolo_loss(detect0, label, de_de0, _ANCHORS[6:9], image_size)
    #     break