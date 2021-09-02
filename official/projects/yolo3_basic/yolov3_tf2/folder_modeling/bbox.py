import tensorflow as tf
from modeling.darknet import ConvModule, BlockModule, Darknet 
from modeling.heads import GenModule, PredModule
import numpy as np

def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(grid_size[1],grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes, max_boxes=100):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs

    dscores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(dscores,[1])
    bbox = tf.reshape(bbox,(-1,4))
    classes = tf.argmax(dscores,1)

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=100,
        iou_threshold=0.5,
        score_threshold=0.5,
        soft_nms_sigma=0.5
    )
    
    num_valid_nms_boxes = tf.shape(selected_indices)[0]
    substract_valid = max_boxes - num_valid_nms_boxes
    selected_indices = tf.concat([selected_indices,tf.zeros(substract_valid, tf.int32)], 0)
    selected_scores = tf.concat([selected_scores,tf.zeros(substract_valid, tf.float32)], -1)

    boxes = tf.gather(bbox, selected_indices)
    boxes = tf.expand_dims(boxes, axis=0)
    scores=selected_scores
    scores = tf.expand_dims(scores, axis=0)
    classes = tf.gather(classes,selected_indices)
    classes = tf.expand_dims(classes, axis=0)
    valid_detections=num_valid_nms_boxes
    valid_detections = tf.expand_dims(valid_detections, axis=0)

    return boxes, scores, classes, valid_detections

# class BBox(tf.keras.layers.Layer):
#     def __init__(self, anchors, num_classes=80):
#         super(BBox, self).__init__()
#         self.anchors = anchors
#         self.num_classes = num_classes

#     def call(self, pred):
#         # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
#         grid_size = tf.shape(pred)[1:3]
#         box_xy, box_wh, objectness, class_probs = tf.split(
#             pred, (2, 2, 1, self.num_classes), axis=-1)

#         box_xy = tf.sigmoid(box_xy)
#         objectness = tf.sigmoid(objectness)
#         class_probs = tf.sigmoid(class_probs)
#         pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

#         # !!! grid[x][y] == (y, x)
#         grid = tf.meshgrid(grid_size[1],grid_size[0])
#         grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

#         box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
#             tf.cast(grid_size, tf.float32)
#         box_wh = tf.exp(box_wh) * self.anchors

#         box_x1y1 = box_xy - box_wh / 2
#         box_x2y2 = box_xy + box_wh / 2
#         bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

#         return bbox, objectness, class_probs, pred_box    


# """

# """
# class NMS(tf.keras.layers.Layer): 
#     def __init__(self, anchors, masks, num_classes=80): # max_boxes=100, iou_threshold=0.5, score_threshold=100, 
#         super(NMS, self).__init__()
#         self.anchors = anchors 
#         self.masks = masks
#         self.num_classes = num_classes

#     def call(self, x_out):
#         # boxes, conf, type
#         b, c, t = [], [], []

#         for o in x_out:
#             b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
#             c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
#             t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

#         bbox = tf.concat(b, axis=1)
#         confidence = tf.concat(c, axis=1)
#         class_probs = tf.concat(t, axis=1)

#         scores = confidence * class_probs

#         dscores = tf.squeeze(scores, axis=0)
#         scores = tf.reduce_max(dscores,[1])
#         bbox = tf.reshape(bbox,(-1,4))
#         classes = tf.argmax(dscores,1)
#         selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
#             boxes=bbox,
#             scores=scores,
#             max_output_size=100,
#             iou_threshold=0.5, 
#             score_threshold=100,
#             soft_nms_sigma=0.5
#         )
        
#         num_valid_nms_boxes = tf.shape(selected_indices)[0]
#         max_boxes_substract_valid = self.max_boxes - num_valid_nms_boxes
#         selected_indices = tf.concat([selected_indices,tf.zeros(max_boxes_substract_valid, tf.int32)], 0)
#         selected_scores = tf.concat([selected_scores,tf.zeros(max_boxes_substract_valid, tf.float32)], -1)

#         boxes = tf.gather(bbox, selected_indices)
#         boxes = tf.expand_dims(boxes, axis=0)

#         scores = tf.expand_dims(selected_scores, axis=0)
#         classes = tf.gather(classes,selected_indices)
#         classes = tf.expand_dims(classes, axis=0)

#         valid_detections = tf.expand_dims(num_valid_nms_boxes, axis=0)

#         return boxes, scores, classes, valid_detections


# Tests
if __name__ == '__main__':
    # test darknet.py
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

    # test heads.py
    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    print('\nTest custom model: GenModule() w/o skip connections')
    gen_m = GenModule(512)
    x = gen_m(x)
    print('Test weights:', len(gen_m.weights))
    print('Test trainable weights:', len(gen_m.trainable_weights))

    print('\nTest custom model: PredModule()')
    pred_m = PredModule(512, len(masks[0]))
    y0 = pred_m(x)
    print('Test weights:', len(pred_m.weights))
    print('Test trainable weights:', len(pred_m.trainable_weights))

    print('\nTest custom model: GenModule() w/ skip connections x_61')
    gen_m = GenModule(256)
    print(x.shape, x_61.shape)
    x = gen_m((x, x_61))
    print('Test weights:', len(gen_m.weights))
    print('Test trainable weights:', len(gen_m.trainable_weights))

    print('\nTest custom model: PredModule()')
    pred_m = PredModule(256, len(masks[1]))
    y1 = pred_m(x)
    print('Test weights:', len(pred_m.weights))
    print('Test trainable weights:', len(pred_m.trainable_weights))

    print('\nTest custom model: GenModule() w/ skip connections x_36')
    gen_m = GenModule(128)
    print(x.shape, x_36.shape)
    x = gen_m((x, x_36))
    print('Test weights:', len(gen_m.weights))
    print('Test trainable weights:', len(gen_m.trainable_weights))

    print('\nTest custom model: PredModule()')
    pred_m = PredModule(128, len(masks[1]))
    y2 = pred_m(x)
    print('Test weights:', len(pred_m.weights))
    print('Test trainable weights:', len(pred_m.trainable_weights))

    # test bbox.py
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
    print('\nTest BBox() w output0')
    bbox_m = BBox(anchors[masks[0]])
    z0 = bbox_m(y0)
    print('Test len z0:', len(z0))
    print('Test z0:', z0)

    print('\nTest BBox() w output1')
    bbox_m = BBox(anchors[masks[1]])
    z1 = bbox_m(y1)
    print('Test len z1:', len(z1))
    print('Test z1:', z1)

    print('\nTest BBox() w output2')
    bbox_m = BBox(anchors[masks[2]])
    z2 = bbox_m(y2)
    print('Test len z2:', len(z2))
    print('Test z2:', z2)

    print('\nTest NMS')
    nms_m = NMS(anchors, masks)
    zz = nms_m((z0[:3], z1[:3], z2[:3]))
    print('Test len zz:', len(zz))
    