
from absl import app, flags, logging
from absl.flags import FLAGS
import os 
import tensorflow as tf
import cv2

from ops.bbox import chunk_anchors
from dataloader.data import load_class_names
from tasks.task import build_model, inference
from utils.interface import draw_boxes_cv2 


flags.DEFINE_string('image', './data/dog.jpg', 'path of video file)')
flags.DEFINE_string('output_folder', './outputs/', 'path to output image')

flags.DEFINE_string('classes','./data/coco.names', 'path of class label text file')
flags.DEFINE_string('pretrain_weights','./data/yolov3.weights', 'path of class label text file')

flags.DEFINE_multi_integer('model_size', (608, 608), 'Resolution of DNN input, must be the multiples of 32')
flags.DEFINE_integer('max_out_size', 20 , 'maximum detected object amount of one class')
flags.DEFINE_float('iou_threshold', 0.4 , 'threshold of non-max suppression') 
flags.DEFINE_float('confid_threshold', 0.3 , 'threshold of confidence') 



_ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]
logging.set_verbosity(logging.INFO)


def main(_argv):
    del _argv

    # setup hardware: GPU and multi-threads
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    
    # prepare anchors and classes
    anchors = chunk_anchors(_ANCHORS, 3)
    classes = load_class_names(FLAGS.classes)
    n_classes= len(classes)

    # build YOLO model and load pre-trained weights
    model = build_model(n_classes, anchors, FLAGS.model_size, FLAGS.pretrain_weights, True)
    
    # parse images for tensor
    raw_img = cv2.imread(FLAGS.image)
    img = tf.expand_dims(raw_img, 0)  # return a tensor
    img = tf.cast(img, dtype=tf.float32)
    img = img / 255.0
    img = tf.image.resize(img, size=FLAGS.model_size)

    # run detection task and prediction boxes
    detections = inference(model, img, anchors, FLAGS.model_size, 
            FLAGS.max_out_size, FLAGS.iou_threshold, FLAGS.confid_threshold)
    
    # draw boxes and write image output
    _ = draw_boxes_cv2(raw_img, detections, classes, FLAGS.model_size, detect_video=False)
    
    input_file = FLAGS.image.split('/')[-1]
    output_path = os.path.join( os.getcwd(), FLAGS.output_folder, 'output_' + input_file)
    cv2.imwrite(output_path, raw_img)
    logging.info(f"Saved output at: {output_path}")

if __name__ == '__main__':
    app.run(main)