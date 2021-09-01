import tensorflow as tf

class BBox(tf.keras.layers.Layer):
    def __init__(self, anchors, num_classes=80):
        super(BBox).__init__()
        self.anchors = anchors
        self.num_classes = num_classes

    def call(self, pred):
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, self.num_classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = tf.meshgrid(grid_size[1],grid_size[0])
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
            tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * self.anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, class_probs, pred_box    


"""

"""
class NMS(tf.keras.layers.Layer): 
    def __init__(self, anchors, masks, num_classes=80): # max_boxes=100, iou_threshold=0.5, score_threshold=100, 
        super(BBox).__init__()
        self.anchors = anchors 
        self.masks = masks
        # self.max_boxes = max_boxes
        # self.iou_threshold = iou_threshold
        # self.score_threshold = score_threshold
        self.num_classes = num_classes

    def call(self, x_out):
        # boxes, conf, type
        b, c, t = [], [], []

        for o in x_out:
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
            score_threshold=100,
            soft_nms_sigma=0.5
        )
        
        num_valid_nms_boxes = tf.shape(selected_indices)[0]
        max_boxes_substract_valid = self.max_boxes - num_valid_nms_boxes
        selected_indices = tf.concat([selected_indices,tf.zeros(max_boxes_substract_valid, tf.int32)], 0)
        selected_scores = tf.concat([selected_scores,tf.zeros(max_boxes_substract_valid, tf.float32)], -1)

        boxes = tf.gather(bbox, selected_indices)
        boxes = tf.expand_dims(boxes, axis=0)

        scores = tf.expand_dims(selected_scores, axis=0)
        classes = tf.gather(classes,selected_indices)
        classes = tf.expand_dims(classes, axis=0)

        valid_detections = tf.expand_dims(num_valid_nms_boxes, axis=0)

        return boxes, scores, classes, valid_detections
