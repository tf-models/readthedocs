
class Yolo3Model(tf.keras.Model):
    """A example model class.
    A model is a subclass of tf.keras.Model where layers are built in the
    constructor.
    """
    def __init__(
        self,
        num_classes: int,
        input_specs: tf.keras.layers.InputSpec = tf.keras.layers.InputSpec(
            shape=[None, None, None, 3]),
        **kwargs):
        """Initializes the example model.
        All layers are defined in the constructor, and config is recorded in the
        `_config_dict` object for serialization.
        Args:
        num_classes: The number of classes in classification task.
        input_specs: A `tf.keras.layers.InputSpec` spec of the input tensor.
        **kwargs: Additional keyword arguments to be passed.
        """

def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')