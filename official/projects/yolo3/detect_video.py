
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import time
import statistics

import tensorflow as tf
import cv2 

from ops.bbox import chunk_anchors
from dataloader.data import load_class_names
from tasks.task import build_model, inference
from utils.interface import draw_boxes_cv2 


flags.DEFINE_boolean('video', True, 'if True, video from the specific path will be used. Otherwise, specific camera will be used')
flags.DEFINE_string('video_path', './data/times_square.mp4', 'path of video file)')
flags.DEFINE_integer('cam_num', 0 , 'number of the camera')
flags.DEFINE_float('prop_fps', 20.0, 'set requency of capture/write frame')

flags.DEFINE_multi_integer('output_res', (1280, 720), 'output resolution of video') # (1920, 1080)
flags.DEFINE_string('output_folder','./outputs/' , 'path folder output video')
# flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')

flags.DEFINE_string('classes','./data/coco.names', 'path of class label text file')
flags.DEFINE_string('pretrain_weights','./data/yolov3.weights', 'path of class label text file')

flags.DEFINE_multi_integer('model_size', (608, 608), 'resolution of DNN input, must be the multiples of 32')
flags.DEFINE_integer('max_out_size', 20 , 'maximum detected object amount of one class')
flags.DEFINE_float('iou_threshold', 0.4 , 'threshold of non-max suppression')
flags.DEFINE_float('confid_threshold', 0.3 , 'threshold of confidence')



_ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]
logging.set_verbosity(logging.INFO)


def main(argv):
    del argv

    # setup hardware: GPU and multi-threads
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    # set up video stream utils
    if FLAGS.video:
        cap = cv2.VideoCapture(FLAGS.video_path)
        output_type = 'video_'
        # output_path = os.path.join( os.getcwd(), FLAGS.output_folder, 'output_' + input_file + '.avi')
        # output_path = FLAGS.output_folder + time.strftime('%m%d%H%M') + input_file + '.avi'
        # print(output_path)
        # for i in output_file:
        #     if '.mp4' in i:
        #         output_file = i.rstrip('.mp4')
        # output_file = FLAGS.output_folder + output_file + '_' +  + '.mp4'

    else:
        cap = cv2.VideoCapture(FLAGS.cam_num)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
        cap.set(cv2.CAP_PROP_FPS, FLAGS.prop_fps)
        output_type = 'camera_'
        # output_file = FLAGS.output_folder + 'CAMERA' + '_' + time.strftime('%m%d%H%M') + '.avi'

    # 
    input_name = FLAGS.video_path.split('/')[-1].split('.')[0] 
    output_path = FLAGS.output_folder + output_type + time.strftime('%m%d%H%M') + '_' + input_name + '.avi'

    # fourcc = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    # out = cv2.VideoWriter(output_file, fourcc, 10, (FLAGS.output_res[0], FLAGS.output_res[1]))
    # output_path = FLAG.output_folder + 'output_' + FLAG.video_path.split('/')[-1] # 'D', 'I', 'V', 'X'  # 

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(output_path, fourcc, FLAGS.prop_fps, FLAGS.output_res)
    if not cap.isOpened():
        logging.error("Cannot get streaming")
        exit()

    # prepare anchors and classes
    anchors = chunk_anchors(_ANCHORS, 3)
    classes = load_class_names(FLAGS.classes)
    n_classes= len(classes)
    model = build_model(n_classes, anchors, FLAGS.model_size, 
                        FLAGS.pretrain_weights, load_full_weights=True)
    logging.info('Video frames are being captured.')

    # run video detection
    fps_list = []
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            break

        # parse images for tensor
        frame = cv2.resize(frame, tuple(FLAGS.output_res))
        frame_tf = tf.expand_dims(frame, axis=0)
        frame_tf = tf.cast(frame_tf, dtype=tf.float32)
        frame_tf = tf.image.resize(frame_tf / 255.0, size=FLAGS.model_size)
        
        # run detection task and prediction boxes
        detections = inference(model, frame_tf, anchors, FLAGS.model_size, 
                FLAGS.max_out_size, FLAGS.iou_threshold, FLAGS.confid_threshold)

        # draw boxes, write on frame, return fps info
        fps = draw_boxes_cv2(frame, detections, classes, FLAGS.model_size, detect_video=True)
        fps_list.append(fps)
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('YOLO frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    average_fps = statistics.mean(fps_list)
    logging.info('Average FPS is {:.1f}'.format(average_fps))


if __name__ == '__main__':
    app.run(main)