import cv2 
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import time
import statistics

from utils.interface import inference, load_model, draw_boxes_cv2 
from ops.bbox import chunk_anchors
from dataloader.data import load_class_names



# flags.DEFINE_boolean('video', False,
#                      'If True, video from the specific path will be used. Otherwise, specific camera will be used')
flags.DEFINE_string('image', './data/dog.jpg',
                    'path of video file)')
# flags.DEFINE_integer('cam_num', 0 , 'number of the camera')


flags.DEFINE_multi_integer('output_res', (1920, 1080), 'output resolution of video')
flags.DEFINE_string('output_folder','./output/' , 'path folder output video')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
# flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')

# flags.DEFINE_integer('image_size', 416, 'resize images to')


flags.DEFINE_multi_integer('model_size', (608, 608), 'Resolution of DNN input, must be the multiples of 32')
flags.DEFINE_integer('max_out_size', 20 , 'maximum detected object amount of one class')
flags.DEFINE_float('iou_threshold', 0.4 , 'threshold of non-max suppression') # 0.4
flags.DEFINE_float('confid_threshold', 0.3 , 'threshold of confidence') # 0.3
flags.DEFINE_string('classes','./data/coco.names', 'path of class label text file')
flags.DEFINE_string('pretrain_weights','./data/yolov3.weights', 'path of class label text file')

_ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]
logging.set_verbosity(logging.INFO)

def main(argv):
    del argv

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    anchors = chunk_anchors(_ANCHORS, 3)

    classes = load_class_names(FLAGS.classes)
    n_classes= len(classes)
    model = load_model(n_classes, anchors, FLAGS.model_size, FLAGS.pretrain_weights, True)
    
    raw_img = cv2.imread(FLAGS.image)
    img = tf.expand_dims(raw_img, 0)
    img = tf.cast(img, dtype=tf.float32)
    img = img / 255.0
    img = tf.image.resize(img, size=FLAGS.model_size)

    detections = inference(model, img, anchors, FLAGS.model_size, FLAGS.max_out_size, FLAGS.iou_threshold, FLAGS.confid_threshold)
    _ = draw_boxes_cv2(raw_img, detections, classes, FLAGS.model_size, detect_video=False)
    cv2.imwrite(FLAGS.output, raw_img)

if __name__ == '__main__':
    app.run(main)