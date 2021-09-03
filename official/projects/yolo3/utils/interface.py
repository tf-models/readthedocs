import tensorflow as tf
import os
from absl import logging
from seaborn import color_palette
import numpy as np
import cv2 
import time

from modeling.yolo3 import YOLOV3
from dataloader.load_weight import load_weights
from ops.bbox import non_max_suppression, build_boxes
from ops.decode import decode 


def load_model(num_classes, anchors, MODEL_SIZE, PRETRAIN_WEIGHTS_PATH, load_full_weights):

    model = YOLOV3(n_classes=num_classes, anchors=anchors)
    model.build(input_shape=(None, MODEL_SIZE[0], MODEL_SIZE[1], 3))
    dir = os.path.join( os.getcwd(), PRETRAIN_WEIGHTS_PATH )
    load_weights(model.variables, file=dir, load_full_weights=load_full_weights)
    logging.info('Weights are loaded.')

    return model

@tf.function
def inference(model, inputs, anchors, model_size, max_output_size, iou_threshold, confidence_threshold):
    detect0, detect1, detect2 = model(inputs, training=False)
    de_detect0, de_detect1, de_detect2 = decode(detect0, anchors[2], model.n_classes, model_size), \
                                         decode(detect1, anchors[1], model.n_classes, model_size), \
                                         decode(detect2, anchors[0], model.n_classes, model_size)
    x = tf.concat([de_detect0, de_detect1, de_detect2], axis=1)
    x = build_boxes(x)
    boxes_dicts = non_max_suppression(x, model.n_classes, max_output_size, iou_threshold, confidence_threshold)
    return boxes_dicts


"""
draw boxes
"""

def draw_boxes_cv2(img, boxes_dicts, class_names, model_size, detect_video=False):
    
    time_begin = time.time()
    boxes_dicts = boxes_dicts[0]
    colors = (np.array(color_palette("hls", 80)) * 255).astype(np.uint8) 
    fontface = cv2.FONT_HERSHEY_DUPLEX 
    resize_factor = (img.shape[1] / model_size[0], img.shape[0] / model_size[1])

    for cls in range(len(class_names)):
        boxes = boxes_dicts[cls]
        if np.size(boxes) != 0:
            color = tuple(int(i) for i in colors[cls])
            for box in boxes:
                # thickness of text and rectangle
                thickness = 2
                # box position, confidence score
                xy, confidence = box[:4], box[4]
                # replace non-finite tensor element
                xy = tf.where(tf.math.is_finite(xy), xy, tf.zeros_like(xy))
                # positions for drawing rectangle 
                xy = [xy[i].numpy() * resize_factor[i % 2] for i in range(4)]
                x0, y0 = int(xy[0]) - thickness, int(xy[1]) - thickness
                x1, y1 = int(xy[2]) + thickness, int(xy[3]) + thickness
                # draw object boxes
                cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness) # , lineType=cv.LINE_4
                text_prob = '{} {:.2f}%'.format(class_names[cls], confidence.numpy() * 100)
                textsize = cv2.getTextSize(text_prob, fontFace=fontface, fontScale=0.5, thickness=thickness)
                # draw text boxes
                cv2.rectangle(img, (x0, y0), (x0 + textsize[0][0], y0 - textsize[0][1]),color=color, thickness=-1)
                cv2.putText(img, text_prob, org=(x0, y0), fontFace=fontface, fontScale=0.5, color=(255,255,255), lineType=cv2.LINE_AA)

    if detect_video:
        # put fps on top-left corner
        fps = 1 / (time.time() - time_begin)
        text_time = '{:.1f} fps'.format(fps)
        cv2.putText(img, text_time, org=(10, 20), fontFace=fontface, fontScale=0.5, color=(255, 255, 255), lineType=cv2.LINE_AA)
        return fps

    return 




