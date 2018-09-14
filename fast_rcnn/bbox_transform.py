# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from fast_rcnn.config import cfg
from utils.timer import Timer
import numpy.random as npr
from utils.mylabs import getHeatmapBygkern

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def keyPoints_transform(ex_rois, gt_keyPoints, ex_rois_keyPoints=None, strategy='center'):
    ex_widths = np.array(ex_rois[:, 2] - ex_rois[:, 0] + 1.0)
    ex_heights = np.array(ex_rois[:, 3] - ex_rois[:, 1] + 1.0)
    ex_ctr_x = np.array(ex_rois[:, 0] + 0.5 * ex_widths)
    ex_ctr_y = np.array(ex_rois[:, 1] + 0.5 * ex_heights)
    keyPoint_num = gt_keyPoints.shape[1] / 2
    keyPoints = gt_keyPoints.reshape((-1, 2))

    # if cfg.TRAIN.VISUAL_ANCHORS_IMG_Flipped:
    #     print 'flip'
    if strategy == 'center':
        targets_dx = (keyPoints[:, 0] - ex_ctr_x.repeat(keyPoint_num)) / ex_widths.repeat(keyPoint_num)
        targets_dy = (keyPoints[:, 1] - ex_ctr_y.repeat(keyPoint_num)) / ex_heights.repeat(keyPoint_num)
    elif strategy == 'left':
        targets_dx = (keyPoints[:, 0] - np.array(ex_rois[:, 0]).repeat(keyPoint_num)) / ex_widths.repeat(keyPoint_num)
        targets_dy = (keyPoints[:, 1] - np.array(ex_rois[:, 1]).repeat(keyPoint_num)) / ex_heights.repeat(keyPoint_num)
    elif strategy == '3_center':
        targets_dx = (keyPoints[:, 0] - ex_ctr_x.repeat(keyPoint_num)) / ex_widths.repeat(keyPoint_num)
        targets_dy = (keyPoints[:, 1] - ex_ctr_y.repeat(keyPoint_num)) / ex_heights.repeat(keyPoint_num)
        targets_dx = targets_dx.reshape(-1, keyPoint_num)
        targets_dy = targets_dy.reshape(-1, keyPoint_num)
        keyPoints_dx = keyPoints[:, 0].reshape(-1, keyPoint_num)
        keyPoints_dy = keyPoints[:, 1].reshape(-1, keyPoint_num)
        ex_lctr_x = ex_rois[:, 0].repeat(9).reshape(-1, 9)
        ex_lctr_y = ex_ctr_y[:, 0].repeat(9).reshape(-1, 9)
        ex_widths = ex_widths.repeat(9).reshape(-1, 9)
        ex_heights = ex_heights.repeat(9).reshape(-1, 9)
        targets_dx[:, 1:10] = (keyPoints_dx[:, 1:10] - ex_lctr_x) / ex_widths
        targets_dy[:, 1:10] = (keyPoints_dy[:, 1:10] - ex_lctr_y) / ex_heights
    elif strategy == 'elewise':
        assert ex_rois_keyPoints is not None
        ex_rois_keyPoints = ex_rois_keyPoints.reshape((-1, 2))
        targets_dx = (keyPoints[:, 0] - ex_rois_keyPoints[:, 0]) / ex_widths.repeat(keyPoint_num)
        targets_dy = (keyPoints[:, 1] - ex_rois_keyPoints[:, 1]) / ex_heights.repeat(keyPoint_num)
    targets = np.hstack(
        (targets_dx[:, np.newaxis], targets_dy[:, np.newaxis])).reshape((-1, keyPoint_num*2))
    # reset value of invalid keyPoints (0, 1)
    invalid_kp = _get_invalid_keyPoint_ins(keyPoints)
    targets[invalid_kp, :] = np.array([0, 1]).repeat(keyPoint_num)
    return targets


def _get_invalid_keyPoint_ins(keyPoints):
    keyPoints = keyPoints.reshape(-1, cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'])
    invalid_kp = []
    for i, keyPoint in enumerate(keyPoints):
        if len(set(keyPoint)) <= 2:
            invalid_kp.append(i)
    return invalid_kp


def keyPoints_transform_inv(ex_rois, deltas, ex_rois_keyPoints=None, strategy='center'):
    if ex_rois.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    ex_rois = ex_rois.astype(deltas.dtype, copy=False)

    widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ctr_x = ex_rois[:, 0] + 0.5 * widths
    ctr_y = ex_rois[:, 1] + 0.5 * heights

    keyPoint_num = deltas.shape[1] / 2
    keyPoint_deltas = deltas.reshape((-1, 2))

    if strategy == 'center':
        pre_dx = (keyPoint_deltas[:, 0] * widths.repeat(keyPoint_num) + ctr_x.repeat(keyPoint_num))
        pre_dy = (keyPoint_deltas[:, 1] * heights.repeat(keyPoint_num) + ctr_y.repeat(keyPoint_num))
    elif strategy == 'left':
        pre_dx = (keyPoint_deltas[:, 0] * widths.repeat(keyPoint_num) + ex_rois[:, 0].repeat(keyPoint_num))
        pre_dy = (keyPoint_deltas[:, 1] * heights.repeat(keyPoint_num) + ex_rois[:, 1].repeat(keyPoint_num))
    elif strategy == 'elewise':
        assert ex_rois_keyPoints is not None
        ex_rois_keyPoints = ex_rois_keyPoints.reshape((-1, 2))
        pre_dx = (keyPoint_deltas[:, 0] * widths.repeat(keyPoint_num) + ex_rois_keyPoints[:, 0])
        pre_dy = (keyPoint_deltas[:, 1] * heights.repeat(keyPoint_num) + ex_rois_keyPoints[:, 1])

    pres = np.hstack(
        (pre_dx[:, np.newaxis], pre_dy[:, np.newaxis])).reshape((-1, keyPoint_num*2))
    return pres


def kp_map_transform_reg(ex_roi, gt_keyPoint, offset_scale):
    ex_roi_t = ex_roi.copy()
    kp_map = np.zeros((gt_keyPoint.shape[0]/2, cfg.TRAIN.MAP_SIZE, cfg.TRAIN.MAP_SIZE))  # bg
    ex_roi_w = ex_roi[2] - ex_roi[0]
    ex_roi_h = ex_roi[3] - ex_roi[1]
    ex_roi_t_w = ex_roi_w * offset_scale
    ex_roi_t_h = ex_roi_h * offset_scale
    ex_roi_t[0] = ex_roi[0] - (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[2] = ex_roi[2] + (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[1] = ex_roi[1] - (ex_roi_t_h - ex_roi_h) / 2
    ex_roi_t[3] = ex_roi[3] + (ex_roi_t_h - ex_roi_h) / 2
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_h
    gt_keyPoint = np.array(gt_keyPoint).reshape((-1, 2))
    gt_keyPoint[:, 0] = (gt_keyPoint[:, 0] - ex_roi_t[0]) * roi_map_x_scale
    gt_keyPoint[:, 1] = (gt_keyPoint[:, 1] - ex_roi_t[1]) * roi_map_y_scale
    valids_i = np.where((gt_keyPoint[:, 0] >= 0) &
                       (gt_keyPoint[:, 1] >= 0) &
                       (gt_keyPoint[:, 0] <= cfg.TRAIN.MAP_SIZE - 1) &
                       (gt_keyPoint[:, 1] <= cfg.TRAIN.MAP_SIZE - 1))[0]
    if cfg.TRAIN.KP_MAP_SAMPLE == 0:
        for valid_i in valids_i:
            kp_map[valid_i, np.round(gt_keyPoint[valid_i, 1]).astype(np.int),
                   np.round(gt_keyPoint[valid_i, 0]).astype(np.int)] = 1
    elif cfg.TRAIN.KP_MAP_SAMPLE == 1:
        kp_map.fill(-1)
        for valid_i in valids_i:
            kp_map[valid_i, np.round(gt_keyPoint[valid_i, 1]).astype(np.int),
                   np.round(gt_keyPoint[valid_i, 0]).astype(np.int)] = 1
        fg_num = len(valids_i)
        bg_num = int(fg_num / cfg.TRAIN.KP_MAP_FG_FRACTION * (1 - cfg.TRAIN.KP_MAP_FG_FRACTION))
        for valid_i in valids_i:
            kp_map_i = kp_map[valid_i, :, :].ravel()
            bg_inds = np.where(kp_map_i == -1)[0]
            if bg_inds.size > 0:
                bg_inds = npr.choice(bg_inds, size=bg_num, replace=False)
            kp_map_i[bg_inds] = 0
            kp_map[valid_i, :, :] = kp_map_i.reshape(cfg.TRAIN.MAP_SIZE, cfg.TRAIN.MAP_SIZE)
    elif cfg.TRAIN.KP_MAP_SAMPLE == 2:
        pass

    if cfg.TRAIN.KP_REGRESSION_Gaussian:
        kp_map = getHeatmapBygkern(kp_map, kernlen=cfg.TRAIN.Gaussian_kernlen,
                                   nsig=cfg.TRAIN.Gaussian_sigma)
    return kp_map


def kp_map_transform_inv_reg(ex_roi, kp_map, offset_scale):
    ex_roi_t = ex_roi.copy()
    assert kp_map.shape[1] == cfg.TRAIN.MAP_SIZE
    ex_roi_w = ex_roi[2] - ex_roi[0]
    ex_roi_h = ex_roi[3] - ex_roi[1]
    ex_roi_t_w = ex_roi_w * offset_scale
    ex_roi_t_h = ex_roi_h * offset_scale
    ex_roi_t[0] = ex_roi[0] - (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[2] = ex_roi[2] + (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[1] = ex_roi[1] - (ex_roi_t_h - ex_roi_h) / 2
    ex_roi_t[3] = ex_roi[3] + (ex_roi_t_h - ex_roi_h) / 2
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_h

    kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] / 2
    gt_keyPoint = np.zeros((kp_num, 2))

    for i in range(0, kp_num):
        kp_map_index = np.where(kp_map[i] == 1)
        if len(kp_map_index[0]) != 0:
            gt_keyPoint[i] = [kp_map_index[1] / roi_map_x_scale + ex_roi_t[0],
                                kp_map_index[0] / roi_map_y_scale + ex_roi_t[1]]
    return gt_keyPoint.ravel()


def kp_map_transform_inv_reg_bg(ex_roi, kp_map, offset_scale):
    ex_roi_t = ex_roi.copy()
    assert kp_map.shape[1] == cfg.TRAIN.MAP_SIZE
    ex_roi_w = ex_roi[2] - ex_roi[0]
    ex_roi_h = ex_roi[3] - ex_roi[1]
    ex_roi_t_w = ex_roi_w * offset_scale
    ex_roi_t_h = ex_roi_h * offset_scale
    ex_roi_t[0] = ex_roi[0] - (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[2] = ex_roi[2] + (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[1] = ex_roi[1] - (ex_roi_t_h - ex_roi_h) / 2
    ex_roi_t[3] = ex_roi[3] + (ex_roi_t_h - ex_roi_h) / 2
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_h

    kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] / 2
    gt_keyPoint = np.zeros((kp_num, 2))

    for i in range(0, kp_num):
        kp_map_index = np.where(kp_map[i] == 1)
        if len(kp_map_index[0]) != 0:
            gt_keyPoint[i] = [kp_map_index[1] / roi_map_x_scale + ex_roi_t[0],
                                kp_map_index[0] / roi_map_y_scale + ex_roi_t[1]]

    gt_keyPoint_bgs = np.array([], dtype=np.float).reshape(2, 0)
    for i in range(0, kp_num):
        kp_map_bg_index = np.where(kp_map[i] == 0)
        if len(kp_map_index[0]) != 0:
            gt_keyPoint_bg = np.array([kp_map_bg_index[1] / roi_map_x_scale + ex_roi_t[0],
                                       kp_map_bg_index[0] / roi_map_y_scale + ex_roi_t[1]])
            gt_keyPoint_bgs = np.hstack([gt_keyPoint_bgs, gt_keyPoint_bg])

    gt_keyPoint_bgs = gt_keyPoint_bgs.transpose()

    return gt_keyPoint.ravel(), gt_keyPoint_bgs.ravel()


def kp_map_transform_cls(ex_roi, gt_keyPoint, offset_scale):
    ex_roi_t = ex_roi.copy()
    kp_map = np.zeros((cfg.TRAIN.MAP_SIZE, cfg.TRAIN.MAP_SIZE))  # bg
    ex_roi_w = ex_roi[2] - ex_roi[0]
    ex_roi_h = ex_roi[3] - ex_roi[1]
    ex_roi_t_w = ex_roi_w * offset_scale
    ex_roi_t_h = ex_roi_h * offset_scale
    ex_roi_t[0] = ex_roi[0] - (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[2] = ex_roi[2] + (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[1] = ex_roi[1] - (ex_roi_t_h - ex_roi_h) / 2
    ex_roi_t[3] = ex_roi[3] + (ex_roi_t_h - ex_roi_h) / 2
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_h
    gt_keyPoint = np.array(gt_keyPoint).reshape((-1, 2))
    gt_keyPoint[:, 0] = (gt_keyPoint[:, 0] - ex_roi_t[0]) * roi_map_x_scale
    gt_keyPoint[:, 1] = (gt_keyPoint[:, 1] - ex_roi_t[1]) * roi_map_y_scale
    valid_i = np.where((gt_keyPoint[:, 0] >= 0) &
                       (gt_keyPoint[:, 1] >= 0) &
                       (gt_keyPoint[:, 0] <= cfg.TRAIN.MAP_SIZE - 1) &
                       (gt_keyPoint[:, 1] <= cfg.TRAIN.MAP_SIZE - 1))[0]
    if not cfg.TRAIN.MAP_ONLY_FG:
        if cfg.TRAIN.KP_MAP_SAMPLE == 0:
            kp_map[np.round(gt_keyPoint[valid_i, 1]).astype(np.int),
                   np.round(gt_keyPoint[valid_i, 0]).astype(np.int)] = valid_i + 1
        elif cfg.TRAIN.KP_MAP_SAMPLE == 1:
            kp_map.fill(-1)
            kp_map[np.round(gt_keyPoint[valid_i, 1]).astype(np.int),
                   np.round(gt_keyPoint[valid_i, 0]).astype(np.int)] = valid_i + 1
            fg_num = len(valid_i)
            bg_num = int(fg_num / cfg.TRAIN.KP_MAP_FG_FRACTION * (1 - cfg.TRAIN.KP_MAP_FG_FRACTION))
            kp_map = kp_map.ravel()
            bg_inds = np.where(kp_map == -1)[0]
            if bg_inds.size > 0:
                bg_inds = npr.choice(bg_inds, size=bg_num, replace=False)
            kp_map[bg_inds] = 0
            kp_map = kp_map.reshape(cfg.TRAIN.MAP_SIZE, cfg.TRAIN.MAP_SIZE)
        elif cfg.TRAIN.KP_MAP_SAMPLE == 2:
            kp_map.fill(-1)

            kp_map_bg_x = np.round(gt_keyPoint[valid_i, 1]).astype(np.int) + 1
            kp_map_bg_y = np.round(gt_keyPoint[valid_i, 0]).astype(np.int)
            kp_map_bg_valid_x_i = np.where(kp_map_bg_x <= cfg.TRAIN.MAP_SIZE - 1)[0]
            kp_map_bg_valid_x = kp_map_bg_x[kp_map_bg_valid_x_i]
            kp_map_bg_valid_y = kp_map_bg_y[kp_map_bg_valid_x_i]
            kp_map[kp_map_bg_valid_x, kp_map_bg_valid_y] = 0

            kp_map_bg_x = np.round(gt_keyPoint[valid_i, 1]).astype(np.int) - 1
            kp_map_bg_valid_x_i = np.where(kp_map_bg_x >= 0)[0]
            kp_map_bg_valid_x = kp_map_bg_x[kp_map_bg_valid_x_i]
            kp_map_bg_valid_y = kp_map_bg_y[kp_map_bg_valid_x_i]
            kp_map[kp_map_bg_valid_x, kp_map_bg_valid_y] = 0

            kp_map_bg_x = np.round(gt_keyPoint[valid_i, 1]).astype(np.int)
            kp_map_bg_y = np.round(gt_keyPoint[valid_i, 0]).astype(np.int) + 1
            kp_map_bg_valid_y_i = np.where(kp_map_bg_y <= cfg.TRAIN.MAP_SIZE - 1)[0]
            kp_map_bg_valid_x = kp_map_bg_x[kp_map_bg_valid_y_i]
            kp_map_bg_valid_y = kp_map_bg_y[kp_map_bg_valid_y_i]
            kp_map[kp_map_bg_valid_x, kp_map_bg_valid_y] = 0

            kp_map_bg_y = np.round(gt_keyPoint[valid_i, 0]).astype(np.int) - 1
            kp_map_bg_valid_y_i = np.where(kp_map_bg_y >= 0)[0]
            kp_map_bg_valid_x = kp_map_bg_x[kp_map_bg_valid_y_i]
            kp_map_bg_valid_y = kp_map_bg_y[kp_map_bg_valid_y_i]
            kp_map[kp_map_bg_valid_x, kp_map_bg_valid_y] = 0

            kp_map[np.round(gt_keyPoint[valid_i, 1]).astype(np.int),
                   np.round(gt_keyPoint[valid_i, 0]).astype(np.int)] = valid_i + 1
    else:
        kp_map.fill(-1)
        kp_map[np.round(gt_keyPoint[valid_i, 1]).astype(np.int),
               np.round(gt_keyPoint[valid_i, 0]).astype(np.int)] = valid_i
    return kp_map


def kp_map_transform_inv_cls(ex_roi, kp_map, offset_scale, re_dict=None):
    ex_roi_t = ex_roi.copy()
    assert kp_map.shape[0] == cfg.TRAIN.MAP_SIZE
    ex_roi_w = ex_roi[2] - ex_roi[0]
    ex_roi_h = ex_roi[3] - ex_roi[1]
    ex_roi_t_w = ex_roi_w * offset_scale
    ex_roi_t_h = ex_roi_h * offset_scale
    ex_roi_t[0] = ex_roi[0] - (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[2] = ex_roi[2] + (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[1] = ex_roi[1] - (ex_roi_t_h - ex_roi_h) / 2
    ex_roi_t[3] = ex_roi[3] + (ex_roi_t_h - ex_roi_h) / 2
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_h

    kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] / 2
    gt_keyPoint = np.zeros((kp_num, 2))

    if not cfg.TRAIN.MAP_ONLY_FG:
        for i in range(1, kp_num + 1):
            kp_map_index = np.where(kp_map == i)
            if len(kp_map_index[0]) != 0:
                gt_keyPoint[i-1] = [kp_map_index[1] / roi_map_x_scale + ex_roi_t[0],
                                    kp_map_index[0] / roi_map_y_scale + ex_roi_t[1]]
            else:
                if re_dict is not None:
                    gt_keyPoint[i - 1] = [re_dict[i][1] / roi_map_x_scale + ex_roi_t[0],
                                          re_dict[i][0] / roi_map_y_scale + ex_roi_t[1]]
    else:
        for i in range(0, kp_num):
            kp_map_index = np.where(kp_map == i)
            if len(kp_map_index[0]) != 0:
                gt_keyPoint[i] = [kp_map_index[1] / roi_map_x_scale + ex_roi_t[0],
                                    kp_map_index[0] / roi_map_y_scale + ex_roi_t[1]]

    return gt_keyPoint.ravel()


def bf_map_transform_inv_cls(ex_roi, kp_map, offset_scale):
    ex_roi_t = ex_roi.copy()
    assert kp_map.shape[0] == cfg.TRAIN.MAP_SIZE
    ex_roi_w = ex_roi[2] - ex_roi[0]
    ex_roi_h = ex_roi[3] - ex_roi[1]
    ex_roi_t_w = ex_roi_w * offset_scale
    ex_roi_t_h = ex_roi_h * offset_scale
    ex_roi_t[0] = ex_roi[0] - (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[2] = ex_roi[2] + (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[1] = ex_roi[1] - (ex_roi_t_h - ex_roi_h) / 2
    ex_roi_t[3] = ex_roi[3] + (ex_roi_t_h - ex_roi_h) / 2
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_h

    kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] / 2
    gt_keyPoint = np.zeros((kp_num, 2))

    kp_map_index = np.where(kp_map > 0)
    if len(kp_map_index[0]) != 0:
        gt_keyPoint = np.array([kp_map_index[1] / roi_map_x_scale + ex_roi_t[0],
                            kp_map_index[0] / roi_map_y_scale + ex_roi_t[1]])
        gt_keyPoint = gt_keyPoint.transpose()
    return gt_keyPoint.ravel()


def kp_map_transform_inv_cls_bg(ex_roi, kp_map, offset_scale):
    ex_roi_t = ex_roi.copy()
    assert kp_map.shape[0] == cfg.TRAIN.MAP_SIZE
    ex_roi_w = ex_roi[2] - ex_roi[0]
    ex_roi_h = ex_roi[3] - ex_roi[1]
    ex_roi_t_w = ex_roi_w * offset_scale
    ex_roi_t_h = ex_roi_h * offset_scale
    ex_roi_t[0] = ex_roi[0] - (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[2] = ex_roi[2] + (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[1] = ex_roi[1] - (ex_roi_t_h - ex_roi_h) / 2
    ex_roi_t[3] = ex_roi[3] + (ex_roi_t_h - ex_roi_h) / 2
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_t_h

    kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] / 2
    gt_keyPoint = np.zeros((kp_num, 2))

    if not cfg.TRAIN.MAP_ONLY_FG:
        for i in range(1, kp_num + 1):
            kp_map_index = np.where(kp_map == i)
            if len(kp_map_index[0]) != 0:
                gt_keyPoint[i-1] = [kp_map_index[1] / roi_map_x_scale + ex_roi_t[0],
                                    kp_map_index[0] / roi_map_y_scale + ex_roi_t[1]]

        kp_map_bg_index = np.where(kp_map == 0)
        gt_keyPoint_bg = np.array([kp_map_bg_index[1] / roi_map_x_scale + ex_roi_t[0],
                                   kp_map_bg_index[0] / roi_map_y_scale + ex_roi_t[1]])
        gt_keyPoint_bg = gt_keyPoint_bg.transpose()

    else:
        for i in range(0, kp_num):
            kp_map_index = np.where(kp_map == i)
            if len(kp_map_index[0]) != 0:
                gt_keyPoint[i] = [kp_map_index[1] / roi_map_x_scale + ex_roi_t[0],
                                    kp_map_index[0] / roi_map_y_scale + ex_roi_t[1]]

    return gt_keyPoint.ravel(), gt_keyPoint_bg.ravel()


def kp_maps_to_bf_maps(kp_maps):
    bf_maps = np.zeros(kp_maps.shape)
    for i, kp_map in enumerate(kp_maps):
        x = kp_map.copy().ravel()
        y = np.where(x > 0)[0]
        x[y] = 1
        bf_maps[i] = x.reshape(kp_map.shape)
    return bf_maps


def kp_maps_to_bf_map(kp_map):
    x = kp_map.copy().ravel()
    y = np.where(x > 0)[0]
    x[y] = 1
    bf_map = x.reshape(kp_map.shape)
    return bf_map


def kp_maps_to_region_maps(kp_maps):
    pass





