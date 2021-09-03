
import tensorflow as tf
from seaborn import color_palette
import numpy as np
import cv2 as cv
import time


def replace_non_finite(tensor):
    return tf.where(tf.math.is_finite(tensor), tensor, tf.zeros_like(tensor))


def draw_boxes_cv2(img, boxes_dicts, class_names, model_size, time_begin):

    boxes_dicts = boxes_dicts[0]
    colors = (np.array(color_palette("hls", 80)) * 255).astype(np.uint8)
    fontface = cv.FONT_HERSHEY_COMPLEX
    resize_factor = (img.shape[1] / model_size[0], img.shape[0] / model_size[1])
    for cls in range(len(class_names)):
        boxes = boxes_dicts[cls]
        if np.size(boxes) != 0:
            color = tuple(int(i) for i in colors[cls])
            for box in boxes:
                xy, confidence = box[:4], box[4]
                xy = replace_non_finite(xy)
                xy = [xy[i].numpy() * resize_factor[i % 2] for i in range(4)]
                x0, y0, x1, y1= int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                thickness = int((img.shape[0] + img.shape[1]) // 800 )
                cv.rectangle(img, (x0-thickness, y0-thickness), (x1+thickness, y1+thickness), color, thickness)
                text_prob = '{} {:.1f}%'.format(class_names[cls], confidence.numpy() * 100)
                textsize= cv.getTextSize(text_prob, fontFace=fontface, fontScale=0.5, thickness=1)
                cv.rectangle(img, (x0 - thickness, y0), (x0 + textsize[0][0], y0 - textsize[0][1] - 5),color=color, thickness=-1)
                cv.putText(img, text_prob,org=(x0 - 2 * thickness, y0 - 5), fontFace=fontface, fontScale=0.5, color=(255,255,255))
    fps = 1 / (time.time() - time_begin)
    text_time = '{:.1f} fps'.format(fps)
    cv.putText(img, text_time, org=(10, 20), fontFace=fontface, fontScale=0.5,color=(255, 255, 255))
    return fps



