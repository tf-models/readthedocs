import tensorflow as tf

class BBox(tf.keras.Model):

    def __init__(
        self, 
        anchors, 
        masks, 
        max_boxes, 
        iou_threshold, 
        score_threshold, 
        num_classes=80
        ):
                        
        self.anchors=anchors 
        # self.masks=masks
        self.num_classes=num_classes
        self.max_boxes=max_boxes
        self.iou_threshold=iou_threshold
        self.score_threshold=score_threshold

        self.cur_anchors=[ anchors[masks[i]] for i in len(masks) ]


    def YoloBoxes(self, pred, anchors, name=None):
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        # anchors = self.anchors[self.masks[anchors_mask_i]]
        anchors = self.cur_anchors[anchors]
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, self.num_classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        # grid = _meshgrid(grid_size[1],grid_size[0])
        grid = tf.meshgrid(grid_size[1], grid_size[0])
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, class_probs, pred_box


    def YoloNMS(self, outputs, name=None):
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
        scores = tf.reduce_max(dscores, [1])
        bbox = tf.reshape(bbox, (-1, 4))
        classes = tf.argmax(dscores, 1)

        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            boxes=bbox,
            scores=scores,
            max_output_size=self.max_boxes,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            soft_nms_sigma=0.5
        )
        
        num_valid_nms_boxes = tf.shape(selected_indices)[0]
        max_boxes_subtract_valid_nms = self.max_boxes - num_valid_nms_boxes
        
        selected_indices = tf.concat([selected_indices, tf.zeros(max_boxes_subtract_valid_nms, tf.int32)], 0)
        selected_scores = tf.concat([selected_scores, tf.zeros(max_boxes_subtract_valid_nms, tf.float32)], -1)

        boxes=tf.gather(bbox, selected_indices)
        boxes = tf.expand_dims(boxes, axis=0)

        # scores=selected_scores
        scores = tf.expand_dims(selected_scores, axis=0)
        classes = tf.gather(classes, selected_indices)
        classes = tf.expand_dims(classes, axis=0)

        # valid_detections=num_valid_nms_boxes
        valid_detections = tf.expand_dims(num_valid_nms_boxes, axis=0)

        return boxes, scores, classes, valid_detections



    # def YoloLoss(self, y_true, y_pred, ignore_thresh=0.5):

    #     # 1. transform all pred outputs
    #     # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
    #     pred_box, pred_obj, pred_class, pred_xywh = self.YoloBoxes(y_pred, self.anchors)
    #     pred_xy = pred_xywh[..., 0:2]
    #     pred_wh = pred_xywh[..., 2:4]

    #     # 2. transform all true outputs
    #     # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
    #     true_box, true_obj, true_class_idx = tf.split(
    #         y_true, (4, 1, 1), axis=-1)
    #     true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
    #     true_wh = true_box[..., 2:4] - true_box[..., 0:2]

    #     # give higher weights to small boxes
    #     box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

    #     # 3. inverting the pred box equations
    #     grid_size = tf.shape(y_true)[1]
    #     grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    #     grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    #     true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
    #     true_wh = tf.math.log(true_wh / self.anchors)
    #     true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

    #     # 4. calculate all masks
    #     obj_mask = tf.squeeze(true_obj, -1)
    #     # ignore false positive when iou is over threshold
    #     best_iou = tf.map_fn(
    #         lambda x: tf.reduce_max(
    #             self.broadcastIoU(
    #                 x[0], 
    #                 tf.boolean_mask(
    #                     x[1], 
    #                     tf.cast(x[2], tf.bool)
    #                     )
    #                 ), axis=-1),
    #         (pred_box, true_box, obj_mask), tf.float32)

    #     ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

    #     # 5. calculate all losses
    #     xy_loss = obj_mask * box_loss_scale * \
    #         tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    #     wh_loss = obj_mask * box_loss_scale * \
    #         tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
    #     obj_loss = binary_crossentropy(true_obj, pred_obj)
    #     obj_loss = obj_mask * obj_loss + \
    #         (1 - obj_mask) * ignore_mask * obj_loss
    #     # TODO: use binary_crossentropy instead
    #     class_loss = obj_mask * sparse_categorical_crossentropy(
    #         true_class_idx, pred_class)

    #     # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
    #     xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
    #     wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
    #     obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    #     class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

    #     return xy_loss + wh_loss + obj_loss + class_loss
