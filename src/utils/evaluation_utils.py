"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.31
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: The utils for evaluation
"""

from __future__ import division
import sys
import tqdm

import torch
import numpy as np
from shapely.geometry import Polygon

sys.path.append('../')

from utils.iou_rotated_boxes_utils import iou_rotated_single_vs_multi_boxes, get_corners_3d


def cvt_box_2_polygon(box):
    """
    :param box: an array of shape [4, 2]
    :return: a shapely.geometry.Polygon object
    """
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)


def compute_iou_nms(idx_self, idx_other, polygons, volumes, hs):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    ious = []
    box1 = polygons[idx_self]
    for idx in idx_other:
        box2 = polygons[idx]
        inter_area = box1.intersection(box2).area
        min_h = min(hs[idx], hs[idx_self])
        inter_volume = inter_area * min_h
        iou = inter_volume / (volumes[idx] + volumes[idx_self] - inter_volume + 1e-12)
        ious.append(iou)

    return np.array(ious, dtype=np.float32)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :8]
        pred_scores = output[:, 8]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        if len(annotations) > 0:
            target_labels = annotations[:, 0]
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = iou_rotated_single_vs_multi_boxes(pred_box, target_boxes).max(dim=0)

                if (iou >= iou_threshold) and (box_index not in detected_boxes):
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics


def nms_cpu(boxes, confs, nms_thresh=0.5):
    """
    :param boxes: [num, 6]
    :param confs: [num, num_classes]
    :param nms_thresh:
    :param min_mode:
    :return:
    """
    # order of reduce confidence (high --> low)
    order = confs.argsort()[::-1]

    x, y, z, boxes_h, w, l, im, re = boxes.transpose(1, 0)
    yaw = np.arctan2(im, re)
    boxes_conners = get_corners_3d(x, y, z, boxes_h, w, l, yaw)
    boxes_polygons = [cvt_box_2_polygon(box_[:4, :2]) for box_ in boxes_conners]  # 4 vertices of the box
    boxes_volumes = boxes_h * w * l

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)
        over = compute_iou_nms(idx_self, idx_other, boxes_polygons, boxes_volumes, boxes_h)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(outputs, conf_thresh=0.95, nms_thresh=0.4):
    """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x, y, w, l, im, re, object_conf, class_score, class_pred)
    """
    if type(outputs).__name__ != 'ndarray':
        outputs = outputs.numpy()
    # outputs shape: (batch_size, 22743, 10)
    batch_size = outputs.shape[0]
    # box_array: [batch, num, 6]
    box_array = outputs[:, :, :6]

    # confs: [batch, num, num_classes]
    confs = outputs[:, :, 6:7] * outputs[:, :, 7:]
    obj_confs = outputs[:, :, 6]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = [None for _ in range(batch_size)]

    for i in range(batch_size):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_obj_confs = obj_confs[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        keep = nms_cpu(l_box_array, l_max_conf, nms_thresh=nms_thresh)

        if (keep.size > 0):
            l_box_array = l_box_array[keep, :]
            l_obj_confs = l_obj_confs[keep].reshape(-1, 1)
            l_max_conf = l_max_conf[keep].reshape(-1, 1)
            l_max_id = l_max_id[keep].reshape(-1, 1)
            bboxes_batch[i] = np.concatenate((l_box_array, l_obj_confs, l_max_conf, l_max_id), axis=-1)
    return bboxes_batch


def post_processing_v2(prediction, conf_thresh=0.95, nms_thresh=0.4):
    """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Prediction: x, y, z, h, w, l, im, re, conf, cls
        Returns detections with shape:
            (x, y, z, h, w, l, im, re, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 8] >= conf_thresh]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 8] * image_pred[:, 9:].max(dim=1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 9:].max(dim=1, keepdim=True)
        # detections: (x, y, z, h, w, l, im, re, object_conf, class_score, class_pred)
        detections = torch.cat((image_pred[:, :9].float(), class_confs.float(), class_preds.float()), dim=1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = iou_rotated_single_vs_multi_boxes(detections[0, :8], detections[:, :8]) > nms_thresh
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 8:9]
            # Merge overlapping bboxes by order of confidence
            detections[0, :8] = (weights * detections[invalid, :8]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if len(keep_boxes) > 0:
            output[image_i] = torch.stack(keep_boxes)

    return output
