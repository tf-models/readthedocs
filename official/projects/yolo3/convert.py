import os
os.chdir( '/Users/Gorgeous/2_Works/tf-models-readthedocs/official/projects/yolo3' )
print('[current working directory]', os.getcwd())


from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np

# from yolov3_tf2.models import YoloV3, YoloV3Tiny
from utils import load_darknet_weights
from modeling.yolo3_model import YOLOv3
from modeling.darknet import ConvModule, BlockModule, Darknet # projects.yolo3
from modeling.heads import GenModule, PredModule, PredModel
import tensorflow as tf

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
# flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


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
    'masks': np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]), 
    'anchors': np.array([(10, 13),  (16, 30),   (33, 23), 
                         (30, 61),  (62, 45),   (59, 119), 
                         (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416.0 ,
    'classes': 80
}

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # if FLAGS.tiny:
    #     yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    # else:
    #     yolo = YoloV3(classes=FLAGS.num_classes)

    yolo = YOLOv3(cfg_model=DARKNET53)
    raw_input = tf.ones(shape=(2, 32, 32, 3))
    yolo(raw_input)
    yolo.summary()
    logging.info('model created')

    # load_darknet_weights(yolo, FLAGS.weights)
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

