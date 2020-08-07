"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.31
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for iou calculation of rotated boxes (on GPU)

"""

from __future__ import division
import sys
from math import pi

import torch
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

sys.path.append('../')

from utils.cal_intersection_rotated_boxes import intersection_area, PolyArea2D


def cvt_box_2_polygon(box):
    """
    :param array: an array of shape [num_conners, 2]
    :return: a shapely.geometry.Polygon object
    """
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)


def get_corners_3d_single(x, y, z, h, w, l, yaw):
    """bev image coordinates format - vectorization
    :param x, y, z, h, w, l, yaw: [num_boxes,]
    :return: num_boxes x (x, y, z) of 8 conners
    """
    device = x.device
    box_conners = torch.zeros((8, 3), device=device, dtype=torch.float)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    # front left
    box_conners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[0, 2] = z - h / 2

    # rear left
    box_conners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[1, 2] = z - h / 2

    # rear right
    box_conners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[2, 2] = z - h / 2

    # front right
    box_conners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[3, 2] = z - h / 2

    box_conners[4, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[4, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[4, 2] = z + h / 2

    # rear left
    box_conners[5, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[5, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[5, 2] = z + h / 2

    # rear right
    box_conners[6, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[6, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[6, 2] = z + h / 2

    # front right
    box_conners[7, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[7, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[7, 2] = z + h / 2

    return box_conners


def get_corners_3d(x, y, z, h, w, l, yaw):
    """bev image coordinates format - vectorization
    :param x, y, z, h, w, l, yaw: [num_boxes,]
    :return: num_boxes x (x, y, z) of 8 conners
    """
    device = x.device
    box_conners = torch.zeros((x.size(0), 8, 3), device=device, dtype=torch.float)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # front left
    box_conners[:, 0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 0, 2] = z - h / 2

    # rear left
    box_conners[:, 1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 1, 2] = z - h / 2

    # rear right
    box_conners[:, 2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 2, 2] = z - h / 2

    # front right
    box_conners[:, 3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 3, 2] = z - h / 2

    box_conners[:, 4, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 4, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 4, 2] = z + h / 2

    # rear left
    box_conners[:, 5, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 5, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 5, 2] = z + h / 2

    # rear right
    box_conners[:, 6, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 6, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 6, 2] = z + h / 2

    # front right
    box_conners[:, 7, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 7, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 7, 2] = z + h / 2

    return box_conners


def get_polygons_areas_fix_xyz(boxes, fix_xyz=100.):
    """
    Args:
        box: (num_boxes, 4) --> w, l, im, re
    """
    device = boxes.device
    n_boxes = boxes.size(0)
    x = torch.full(size=(n_boxes,), fill_value=fix_xyz, device=device, dtype=torch.float)
    y = torch.full(size=(n_boxes,), fill_value=fix_xyz, device=device, dtype=torch.float)
    z = torch.full(size=(n_boxes,), fill_value=fix_xyz, device=device, dtype=torch.float)
    h, w, l, im, re = boxes.t()
    yaw = torch.atan2(im, re)
    boxes_conners_3d = get_corners_3d(x, y, z, h, w, l, yaw)
    boxes_polygons = [cvt_box_2_polygon(box_) for box_ in
                      boxes_conners_3d[:, :4, :2]]  # Take (x,y) of the 4 first conners
    boxes_volumes = h * w * l
    low_h = boxes_conners_3d[:, 0, 2]
    high_h = boxes_conners_3d[:, -1, 2]

    return boxes_polygons, boxes_volumes, low_h, high_h


def iou_rotated_boxes_targets_vs_anchors(a_polygons, a_volumes, a_low_hs, a_high_hs, tg_polygons, tg_volumes, tg_low_hs,
                                         tg_high_hs):
    device = a_volumes.device
    num_anchors = len(a_volumes)
    num_targets_boxes = len(tg_volumes)

    ious = torch.zeros(size=(num_anchors, num_targets_boxes), device=device, dtype=torch.float)

    for a_idx in range(num_anchors):
        for tg_idx in range(num_targets_boxes):
            inter_area = a_polygons[a_idx].intersection(tg_polygons[tg_idx]).area
            low_inter_h = max(a_low_hs[a_idx], tg_low_hs[tg_idx])
            high_inter_h = min(a_high_hs[a_idx], tg_high_hs[tg_idx])
            inter_volume = (high_inter_h - low_inter_h) * inter_area
            iou = inter_volume / (a_volumes[a_idx] + tg_volumes[tg_idx] - inter_volume + 1e-16)
            ious[a_idx, tg_idx] = iou

    return ious


def iou_pred_vs_target_boxes(pred_boxes, target_boxes, GIoU=False, DIoU=False, CIoU=False):
    assert pred_boxes.size() == target_boxes.size(), "Unmatch size of pred_boxes and target_boxes"
    device = pred_boxes.device
    n_boxes = pred_boxes.size(0)

    t_x, t_y, t_z, t_h, t_w, t_l, t_im, t_re = target_boxes.t()
    t_yaw = torch.atan2(t_im, t_re)
    t_conners = get_corners_3d(t_x, t_y, t_z, t_h, t_w, t_l, t_yaw)
    t_volumes = t_h * t_w * t_l
    t_low_hs = t_conners[:, 0, 2]
    t_high_hs = t_conners[:, -1, 2]

    p_x, p_y, p_z, p_h, p_w, p_l, p_im, p_re = pred_boxes.t()
    p_yaw = torch.atan2(p_im, p_re)
    p_conners = get_corners_3d(p_x, p_y, p_z, p_h, p_w, p_l, p_yaw)
    p_volumes = p_h * p_w * p_l
    p_low_hs = p_conners[:, 0, 2]
    p_high_hs = p_conners[:, -1, 2]

    ious = []
    giou_loss = torch.tensor([0.], device=device, dtype=torch.float)
    # Thinking to apply vectorization this step
    for box_idx in range(n_boxes):
        p_cons, t_cons = p_conners[box_idx], t_conners[box_idx]
        if not GIoU:
            p_poly, t_poly = cvt_box_2_polygon(p_cons[:4, :2]), cvt_box_2_polygon(t_cons[:4, :2])  # (x, y) of 4 first
            inter_area = p_poly.intersection(t_poly).area
        else:
            inter_area = intersection_area(p_cons[:4, :2], t_cons[:4, :2])

        low_inter_h = max(p_low_hs[box_idx], t_low_hs[box_idx])
        high_inter_h = min(p_high_hs[box_idx], t_high_hs[box_idx])
        inter_h = max(0., high_inter_h - low_inter_h)
        inter_volume = inter_h * inter_area
        union_volume = p_volumes[box_idx] + t_volumes[box_idx] - inter_volume
        iou = inter_volume / (union_volume + 1e-16)

        if GIoU:
            convex_conners = torch.cat((p_cons[:4, :2], t_cons[:4, :2]), dim=0)
            hull = ConvexHull(convex_conners.clone().detach().cpu().numpy())  # done on cpu, just need indices output
            convex_conners = convex_conners[hull.vertices]
            convex_area = PolyArea2D(convex_conners)
            low_convex_h = min(p_low_hs[box_idx], t_low_hs[box_idx])
            high_convex_h = max(p_high_hs[box_idx], t_high_hs[box_idx])
            convex_h = max(0., high_convex_h - low_convex_h)
            convex_volume = convex_h * convex_area
            giou_loss += 1. - (iou - (convex_volume - union_volume) / (convex_volume + 1e-16))
        else:
            giou_loss += 1. - iou

        if DIoU or CIoU:
            raise NotImplementedError

        ious.append(iou)

    return torch.tensor(ious, device=device, dtype=torch.float), giou_loss


def iou_rotated_single_vs_multi_boxes(single_box, multi_boxes):
    """
    :param pred_box: Numpy array
    :param target_boxes: Numpy array
    :return:
    """

    s_x, s_y, s_z, s_h, s_w, s_l, s_im, s_re = single_box
    s_volume = s_h * s_w * s_l
    s_yaw = torch.atan2(s_im, s_re)
    s_conners = get_corners_3d_single(s_x, s_y, s_z, s_h, s_w, s_l, s_yaw)
    s_polygon = cvt_box_2_polygon(s_conners[:4, :2])
    s_low_h = s_conners[0, 2]
    s_high_h = s_conners[-1, 2]

    m_x, m_y, m_z, m_h, m_w, m_l, m_im, m_re = multi_boxes.transpose(1, 0)
    targets_volumes = m_h * m_w * m_l
    m_yaw = torch.atan2(m_im, m_re)
    m_boxes_conners = get_corners_3d(m_x, m_y, m_z, m_h, m_w, m_l, m_yaw)
    m_boxes_polygons = [cvt_box_2_polygon(box_[:4, :2]) for box_ in m_boxes_conners]
    m_boxes_low_hs = m_boxes_conners[:, 0, 2]
    m_boxes_high_hs = m_boxes_conners[:, -1, 2]

    ious = []
    for m_idx in range(multi_boxes.shape[0]):
        inter_area = s_polygon.intersection(m_boxes_polygons[m_idx]).area
        low_inter_h = max(s_low_h, m_boxes_low_hs[m_idx])
        high_inter_h = min(s_high_h, m_boxes_high_hs[m_idx])
        inter_h = max(0., high_inter_h - low_inter_h)
        inter_volume = inter_area * inter_h
        iou_ = inter_volume / (s_volume + targets_volumes[m_idx] - inter_volume + 1e-16)
        ious.append(iou_)

    return torch.tensor(ious, dtype=torch.float)


if __name__ == "__main__":
    import cv2
    import numpy as np

    # Show convex in an image

    img_size = 300
    img = np.zeros((img_size, img_size, 3))
    img = cv2.resize(img, (img_size, img_size))

    box1 = torch.tensor([100, 100, 100, 80, 20, 30, 0], dtype=torch.float).cuda()
    box2 = torch.tensor([100, 100, 100, 40, 20, 30, 0], dtype=torch.float).cuda()

    box1_conners = get_corners_3d_single(box1[0], box1[1], box1[2], box1[3], box1[4], box1[5], box1[6])
    box1_polygon = cvt_box_2_polygon(box1_conners[:4, :2])
    box1_volume = box1[3] * box1[4] * box1[5]

    box2_conners = get_corners_3d_single(box2[0], box2[1], box2[2], box2[3], box2[4], box2[5], box2[6])
    box2_polygon = cvt_box_2_polygon(box2_conners[:4, :2])
    box2_volume = box2[3] * box2[4] * box2[5]

    inter_area = box2_polygon.intersection(box1_polygon).area
    min_h = min(box1[3], box2[3])
    low_inter_h = max(box1_conners[0, 2], box2_conners[0, 2])
    high_inter_h = min(box1_conners[-1, 2], box2_conners[-1, 2])
    inter_h = max(0., high_inter_h - low_inter_h)
    inter_volume = inter_h * inter_area

    union_volume = box1_volume + box2_volume - inter_volume
    iou = inter_volume / (union_volume + 1e-16)

    convex_conners = torch.cat((box1_conners[:4, :2], box2_conners[:4, :2]), dim=0)
    hull = ConvexHull(convex_conners.clone().detach().cpu().numpy())  # done on cpu, just need indices output
    convex_conners = convex_conners[hull.vertices]
    convex_polygon = cvt_box_2_polygon(convex_conners)
    convex_area = convex_polygon.area
    low_convex_h = min(box1_conners[0, 2], box2_conners[0, 2])
    high_convex_h = max(box1_conners[-1, 2], box2_conners[-1, 2])
    convex_h = max(0., high_convex_h - low_convex_h)
    convex_volume = convex_area * convex_h

    giou_loss = 1. - (iou - (convex_volume - union_volume) / (convex_volume + 1e-16))

    print(
        'box1_volume: {:.2f}, box2_volume: {:.2f}, inter_volume: {:.2f}, union_volume: {:.2f}, iou: {:.4f}, convex_volume: {:.4f}, giou_loss: {}'.format(
            box1_volume, box2_volume, inter_volume, union_volume, iou, convex_volume, giou_loss))

    print('intersection_area: {}'.format(intersection_area(box1_conners[:4, :2], box2_conners[:4, :2])))
    print('convex_area using PolyArea2D: {}'.format(PolyArea2D(convex_conners)))

    img = cv2.polylines(img, [box1_conners[:4, :2].cpu().numpy().astype(np.int)], True, (255, 0, 0), 2)
    img = cv2.polylines(img, [box2_conners[:4, :2].cpu().numpy().astype(np.int)], True, (0, 255, 0), 2)
    img = cv2.polylines(img, [convex_conners.cpu().numpy().astype(np.int)], True, (0, 0, 255), 2)

    while True:
        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xff == 27:
            break
