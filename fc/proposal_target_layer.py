# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform, \
    keyPoints_transform, keyPoints_transform_inv, \
    kp_map_transform_cls, kp_map_transform_inv_cls, kp_map_transform_inv_cls_bg, \
    kp_map_transform_reg, kp_map_transform_inv_reg, kp_map_transform_inv_reg_bg, \
    kp_maps_to_bf_maps, kp_maps_to_bf_map, bf_map_transform_inv_cls


from utils.cython_bbox import bbox_overlaps
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from utils.timer import Timer

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5, 1, 1)
        # labels
        # top[1].reshape(1, 1, 1, 1)
        # # bbox_targets
        # top[2].reshape(1, self._num_classes * 4, 1, 1)
        # # bbox_inside_weights
        # top[3].reshape(1, self._num_classes * 4, 1, 1)
        # # bbox_outside_weights
        # top[4].reshape(1, self._num_classes * 4, 1, 1)

        # TODO consider other types of label
        if cfg.TRAIN.USE_ATTRIBUTE:
            self.index_start = 1
            index = self.index_start
            # if cfg.TRAIN.KP == 6:
            for attr in cfg.TRAIN.ATTRIBUTES:
                if attr.keys()[0] == "gt_keyPoints":
                    if not cfg.TRAIN.PREDICE_KP_MAP:
                        top[index].reshape(1, self._num_classes * attr.values()[0], 1, 1)
                        top[index + 1].reshape(1, self._num_classes * attr.values()[0], 1, 1)
                        top[index + 2].reshape(1, self._num_classes * attr.values()[0], 1, 1)
                        index = index + 3
                    else:
                        map_size = cfg.TRAIN.MAP_SIZE
                        if cfg.TRAIN.PREDICE_KP_REGRESSION:
                            top[index].reshape(1, attr.values()[0]/2, map_size, map_size)
                        else:
                            top[index].reshape(1, 1, map_size, map_size)
                            if cfg.TRAIN.WITH_BF_MAP:
                                top[index+1].reshape(1, 1, map_size, map_size)
                else:
                    top[index].reshape(1, 1, 1, 1)
                    index = index + 1


    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data
        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        if cfg.TRAIN.USE_ATTRIBUTE:
            if cfg.TRAIN.KP == 5 or cfg.TRAIN.KP == 6:
                gt_keyPoints = bottom[2].data
                gt_keyPoints = gt_keyPoints.reshape(gt_keyPoints.shape[0], gt_keyPoints.shape[1])
                if cfg.TRAIN.VISUAL_ATTRIBUTES:
                    visual_attribute(cfg.TRAIN.VISUAL_SCALE, gt_boxes, gt_keyPoints)
                if cfg.TRAIN.RPN_KP_REGRESSION:
                    all_rois_keyPoints = bottom[3].data
                    all_rois_keyPoints = np.vstack(
                        (all_rois_keyPoints, np.hstack((zeros, gt_keyPoints)))
                    )
                else:
                    all_rois_keyPoints = None
                if cfg.TRAIN.KP == 6:
                    gt_ages = bottom[3].data.ravel()
                    gt_genders = bottom[4].data.ravel()
                    gt_ethnicity = bottom[5].data.ravel()
                    gt_attributes = {"gt_ages" : gt_ages, "gt_genders" : gt_genders, "gt_ethnicity" : gt_ethnicity}
                else:
                    gt_attributes = None
            elif cfg.TRAIN.KP == 7:
                gt_keyPoints = None
                all_rois_keyPoints = None
                gt_ages = bottom[2].data.ravel()
                gt_genders = bottom[3].data.ravel()
                gt_ethnicity = bottom[4].data.ravel()
                gt_attributes = {"gt_ages": gt_ages, "gt_genders": gt_genders, "gt_ethnicity": gt_ethnicity}

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        rois_per_image = np.inf if cfg.TRAIN.Frozen_BATCH_SIZE == -1 else cfg.TRAIN.Frozen_BATCH_SIZE
        fg_rois_per_image = int(np.round(cfg.TRAIN.Frozen_FG_FRACTION * rois_per_image)) if rois_per_image != float('inf') else np.round(cfg.TRAIN.Frozen_FG_FRACTION * rois_per_image)
        # fix TypeError:

        # Sample rois with classification labels and bounding box regression
        # targets
        # print 'proposal_target_layer:', fg_rois_per_image

        if cfg.TRAIN.USE_OHEM:
            labels, rois, bbox_targets, bbox_inside_weights = _sample_rois_for_ohem(
                all_rois, gt_boxes, fg_rois_per_image,
                rois_per_image, self._num_classes)
        elif cfg.TRAIN.USE_ROHEM:
            fg_rois_per_image = int(np.round(cfg.TRAIN.Frozen_FG_FRACTION * cfg.TRAIN.USE_ROHEM_BATCHSIZE))
            labels, rois, bbox_targets, bbox_inside_weights = _sample_rois_for_rohem(
                all_rois, gt_boxes, fg_rois_per_image,
                cfg.TRAIN.USE_ROHEM_BATCHSIZE, self._num_classes)
        else:
            if cfg.TRAIN.USE_ATTRIBUTE:
                if not cfg.TRAIN.PREDICE_KP_MAP:
                    labels, rois, bbox_targets, bbox_inside_weights, keyPoint_targets, \
                    keyPoint_inside_weights, attribute_targets = _sample_rois_attributes_gen(
                    all_rois, gt_boxes, fg_rois_per_image, rois_per_image, gt_keyPoints,
                    self._num_classes, all_rois_keyPoints, gt_attributes)
                else:
                    # timer = Timer()
                    # timer.tic()
                    if cfg.TRAIN.WITH_BF_MAP:
                        labels, rois, bbox_targets, bbox_inside_weights, keyPoint_targets, \
                        attribute_targets, bf_targets = _sample_rois_attributes_gen_v2_bfmap(
                            all_rois, gt_boxes, fg_rois_per_image, rois_per_image, gt_keyPoints,
                            self._num_classes, all_rois_keyPoints, gt_attributes)
                    else:
                        labels, rois, bbox_targets, bbox_inside_weights, keyPoint_targets, \
                        attribute_targets = _sample_rois_attributes_gen_v2(
                        all_rois, gt_boxes, fg_rois_per_image, rois_per_image, gt_keyPoints,
                        self._num_classes, all_rois_keyPoints, gt_attributes)
                    # timer.toc()
                    # print ('proposal target took {:.3f}s').format(timer.total_time)
            else:
                labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
                all_rois, gt_boxes, fg_rois_per_image,
                rois_per_image, self._num_classes)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        # modified by ywxiong
        rois = rois.reshape((rois.shape[0], rois.shape[1], 1, 1))
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # # classification labels
        # # modified by ywxiong
        # labels = labels.reshape((labels.shape[0], 1, 1, 1))
        # top[1].reshape(*labels.shape)
        # top[1].data[...] = labels
        #
        # # bbox_targets
        # # modified by ywxiong
        # bbox_targets = bbox_targets.reshape((bbox_targets.shape[0], bbox_targets.shape[1], 1, 1))
        # top[2].reshape(*bbox_targets.shape)
        # top[2].data[...] = bbox_targets
        #
        # # bbox_inside_weights
        # # modified by ywxiong
        # bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        # top[3].reshape(*bbox_inside_weights.shape)
        # top[3].data[...] = bbox_inside_weights
        #
        # # bbox_outside_weights
        # # modified by ywxiong
        # bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        # top[4].reshape(*bbox_inside_weights.shape)
        # top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        if cfg.TRAIN.USE_ATTRIBUTE:
            index = self.index_start
            if cfg.TRAIN.KP == 5 or cfg.TRAIN.KP == 6:
                if not cfg.TRAIN.PREDICE_KP_MAP:
                    # keyPoint_targets
                    # modified by ywxiong
                    keyPoint_targets = keyPoint_targets.reshape((keyPoint_targets.shape[0], keyPoint_targets.shape[1], 1, 1))
                    top[index].reshape(*keyPoint_targets.shape)
                    top[index].data[...] = keyPoint_targets

                    # keyPoint_inside_weights
                    # modified by ywxiong
                    keyPoint_inside_weights = keyPoint_inside_weights.reshape((keyPoint_inside_weights.shape[0], keyPoint_inside_weights.shape[1], 1, 1))
                    top[index+1].reshape(*keyPoint_inside_weights.shape)
                    top[index+1].data[...] = keyPoint_inside_weights

                    # keyPoint_outside_weights
                    # modified by ywxiong
                    keyPoint_inside_weights = keyPoint_inside_weights.reshape((keyPoint_inside_weights.shape[0], keyPoint_inside_weights.shape[1], 1, 1))
                    top[index+2].reshape(*keyPoint_inside_weights.shape)
                    top[index+2].data[...] = np.array(keyPoint_inside_weights > 0).astype(np.float32)
                else:
                    if not cfg.TRAIN.PREDICE_KP_REGRESSION:
                        keyPoint_targets = keyPoint_targets.reshape((keyPoint_targets.shape[0], 1, keyPoint_targets.shape[1], keyPoint_targets.shape[2]))
                        top[index].reshape(*keyPoint_targets.shape)
                        top[index].data[...] = keyPoint_targets
                    else:
                        top[index].reshape(*keyPoint_targets.shape)
                        top[index].data[...] = keyPoint_targets.astype(np.float32)

                    if cfg.TRAIN.WITH_BF_MAP:
                        bf_targets = bf_targets.reshape(
                            (bf_targets.shape[0], 1, bf_targets.shape[1], bf_targets.shape[2]))
                        top[index+1].reshape(*bf_targets.shape)
                        top[index+1].data[...] = bf_targets

                if cfg.TRAIN.KP == 6:
                    index = index + 3
                    for attr in cfg.TRAIN.ATTRIBUTES:
                        attr_name = attr.keys()[0]
                        if attr_name != 'gt_keyPoints':
                            attribute_target = attribute_targets[attr_name]
                            attribute_target = attribute_target.reshape((attribute_target.shape[0], 1, 1, 1))
                            top[index].reshape(*attribute_target.shape)
                            top[index].data[...] = attribute_target
                            index = index + 1
            elif cfg.TRAIN.KP == 7:
                for attr in cfg.TRAIN.ATTRIBUTES:
                    attr_name = attr.keys()[0]
                    attribute_target = attribute_targets[attr_name]
                    attribute_target = attribute_target.reshape((attribute_target.shape[0], 1, 1, 1))
                    top[index].reshape(*attribute_target.shape)
                    top[index].data[...] = attribute_target
                    index = index + 1
            elif cfg.TRAIN.KP == 8:
                pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def visual_attribute(scale, gt_boxes, gt_keyPoints, gt_keyPoints_bg=None):
    if cfg.TRAIN.VISUAL_ANCHORS_IMG != '':
        plt.figure()
        ax = plt.subplot(1, 2, 1)
        im = cv2.imread(cfg.TRAIN.VISUAL_ANCHORS_IMG)
        if cfg.TRAIN.VISUAL_ANCHORS_IMG_Flipped:
            im = im[:, ::-1, :]
        ax.imshow(im[:, :, ::-1], aspect='equal')
        ax = plt.subplot(1, 2, 2)
        if cfg.ENABLE_RON:
            im = cv2.resize(im, None, None, fx=scale[0], fy=scale[1],
                            interpolation=cv2.INTER_LINEAR)
        else:
            im = cv2.resize(im, None, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_LINEAR)
        ax.imshow(im[:, :, ::-1], aspect='equal')
        for gt_boxe in gt_boxes:
            gt_boxe_w = gt_boxe[2] - gt_boxe[0]
            gt_boxe_h = gt_boxe[3] - gt_boxe[1]
            rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                            ec='r', fill=False, lw=1.5)
            ax.add_patch(rec)
        # handle missed gt_keyPoints
        gt_keyPoints = np.array(gt_keyPoints).reshape((-1, 2))
        plt.plot(gt_keyPoints[:, 0], gt_keyPoints[:, 1], 'go', ms=1.5, alpha=1)

        if gt_keyPoints_bg is not None:
            gt_keyPoints_bg = np.array(gt_keyPoints_bg).reshape((-1, 2))
            plt.plot(gt_keyPoints_bg[:, 0], gt_keyPoints_bg[:, 1], 'go', color='b', ms=1.5, alpha=1)
        plt.close('all')


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    # print 'proposal_target_layer:', bbox_targets.shape
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = 4 * int(cls)  # fix slice indices must be integers or None or have an __ind
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _get_bbox_regression_labels_atttribute(bbox_target_data, num_classes, gt_keyPoints_data):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    keyPoint_targets = np.zeros((clss.size, cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] * num_classes), dtype=np.float32)
    # print 'proposal_target_layer:', bbox_targets.shape
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    keyPoint_inside_weights = np.zeros(keyPoint_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = 4 * int(cls)  # fix slice indices must be integers or None or have an __ind
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

            keyPoints_num = gt_keyPoints_data.shape[1]
            start = keyPoints_num * int(cls)  # fix slice indices must be integers or None or have an __ind
            end = start + keyPoints_num
            # handle missed gt_keyPoints
            if len(set(gt_keyPoints_data[ind, :])) != 2:
                keyPoint_targets[ind, start:end] = gt_keyPoints_data[ind, :]
                keyPoint_inside_weights[ind, start:end] = keyPoints_num * [1.0]
            # else:
            #     print 'gt_keyPoints are missed'
    return bbox_targets, bbox_inside_weights, keyPoint_targets, keyPoint_inside_weights


def _get_bbox_regression_labels_atttribute_v2(bbox_target_data, num_classes, gt_keyPoints_data):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    keyPoint_targets = np.zeros((clss.size, cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] * num_classes), dtype=np.float32)
    # print 'proposal_target_layer:', bbox_targets.shape
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    keyPoint_inside_weights = np.zeros(keyPoint_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = 4 * int(cls)  # fix slice indices must be integers or None or have an __ind
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

            keyPoints_num = gt_keyPoints_data.shape[1]
            start = keyPoints_num * int(cls)  # fix slice indices must be integers or None or have an __ind
            end = start + keyPoints_num
            # handle missed gt_keyPoints
            if len(set(gt_keyPoints_data[ind, :])) != 2:
                keyPoint_targets[ind, start:end] = gt_keyPoints_data[ind, :]
                keyPoint_inside_weights[ind, start:end] = keyPoints_num * [1.0]
            # else:
            #     print 'gt_keyPoints are missed'
    return bbox_targets, bbox_inside_weights, keyPoint_targets, keyPoint_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _compute_keyPoints_targets(ex_rois, gt_keyPoints, labels=None, gt_boxes=None, ex_rois_keyPoints=None):
    """Compute keyPoints regression targets for an image."""
    assert ex_rois.shape[1] == 4
    assert gt_keyPoints.shape[1] == cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
    if cfg.TRAIN.RPN_KP_REGRESSION:
        assert ex_rois_keyPoints.shape[1] == gt_keyPoints.shape[1]
        targets = keyPoints_transform(ex_rois, gt_keyPoints, ex_rois_keyPoints, 'elewise')
        if 0:
            pre_keyPoints = keyPoints_transform_inv(ex_rois, targets, ex_rois_keyPoints, 'elewise')
            for index in np.where(labels == 1)[0]:
                visual_attribute(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, [gt_boxes[index]], [gt_keyPoints[index]])
                visual_attribute(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, [gt_boxes[index]], [pre_keyPoints[index]])
    else:
        targets = keyPoints_transform(ex_rois, gt_keyPoints)
        if 0:
            pre_keyPoints = keyPoints_transform_inv(ex_rois, targets)
            for index in np.where(labels == 1)[0]:
                visual_attribute(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, [gt_boxes[index]], [gt_keyPoints[index]])
                visual_attribute(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, [gt_boxes[index]], [pre_keyPoints[index]])

    # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    #     # Optionally normalize targets by a precomputed mean and stdev
    #     targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
    #             / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return targets


def _compute_keyPoints_targets_v2(ex_rois, gt_keyPoints, labels=None, gt_boxes=None):
    """Compute keyPoints regression targets for an image."""
    assert gt_keyPoints.shape[1] == cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
    if cfg.TRAIN.RPN_KP_REGRESSION:
        pass
    else:
        # generate Kp target map
        targets = generate_kp_target_map(ex_rois, gt_keyPoints, gt_boxes, labels, cfg.TRAIN.MAP_offset)

    return targets


def kp_map_transform(ex_roi, gt_keyPoint, offset):
    ex_roi_t = ex_roi.copy()
    kp_map = np.zeros((cfg.TRAIN.MAP_SIZE, cfg.TRAIN.MAP_SIZE))  # bg
    ex_roi_t[2:4] = ex_roi[2:4] + offset
    ex_roi_t[0:2] = ex_roi[0:2] - offset
    ex_roi_w = ex_roi_t[2] - ex_roi_t[0]
    ex_roi_h = ex_roi_t[3] - ex_roi_t[1]
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_h
    gt_keyPoint = np.array(gt_keyPoint).reshape((-1, 2))
    gt_keyPoint[:, 0] = (gt_keyPoint[:, 0] - ex_roi_t[0]) * roi_map_x_scale
    gt_keyPoint[:, 1] = (gt_keyPoint[:, 1] - ex_roi_t[1]) * roi_map_y_scale
    valid_i = np.where((gt_keyPoint[:, 0] >= 0) &
                       (gt_keyPoint[:, 1] >= 0) &
                       (gt_keyPoint[:, 0] <= cfg.TRAIN.MAP_SIZE - 1) &
                       (gt_keyPoint[:, 1] <= cfg.TRAIN.MAP_SIZE - 1))[0]

    kp_map[np.round(gt_keyPoint[valid_i, 1]).astype(np.int),
           np.round(gt_keyPoint[valid_i, 0]).astype(np.int)] = valid_i + 1
    return kp_map


def kp_map_transform_inv(ex_roi, kp_map, offset):
    ex_roi_t = ex_roi.copy()
    assert kp_map.shape[0] == cfg.TRAIN.MAP_SIZE
    ex_roi_t[2:4] = ex_roi[2:4] + offset
    ex_roi_t[0:2] = ex_roi[0:2] - offset
    ex_roi_w = ex_roi_t[2] - ex_roi_t[0]
    ex_roi_h = ex_roi_t[3] - ex_roi_t[1]
    roi_map_x_scale = cfg.TRAIN.MAP_SIZE / ex_roi_w
    roi_map_y_scale = cfg.TRAIN.MAP_SIZE / ex_roi_h

    kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] / 2
    gt_keyPoint = np.zeros((kp_num, 2))
    for i in range(1, kp_num + 1):
        kp_map_index = np.where(kp_map == i)
        if len(kp_map_index[0]) != 0:
            gt_keyPoint[i-1] = [kp_map_index[1] / roi_map_x_scale + ex_roi_t[0],
                                kp_map_index[0] / roi_map_y_scale + ex_roi_t[1]]

    return gt_keyPoint.ravel()


def generate_kp_target_map(ex_rois, gt_keyPoints, gt_boxes, labels, offset):
    if cfg.TRAIN.PREDICE_KP_REGRESSION:
        kp_target_maps = np.zeros((len(labels), gt_keyPoints.shape[1]/2, cfg.TRAIN.MAP_SIZE, cfg.TRAIN.MAP_SIZE))
    else:
        kp_target_maps = np.zeros((len(labels), cfg.TRAIN.MAP_SIZE, cfg.TRAIN.MAP_SIZE))
    kp_target_maps.fill(-1)
    for i, ex_roi in enumerate(ex_rois):
        if labels[i]:
            if cfg.TRAIN.PREDICE_KP_REGRESSION:
                kp_target_maps[i] = kp_map_transform_reg(ex_roi, gt_keyPoints[i], offset)
            else:
                kp_target_maps[i] = kp_map_transform_cls(ex_roi, gt_keyPoints[i], offset)
            if 0:  # debug
                visual_attribute(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, [ex_roi], [gt_keyPoints[i]])
                temp_size = cfg.TRAIN.MAP_SIZE
                for size in [75, 90, 105, 120]:
                    cfg.TRAIN.MAP_SIZE = size
                    kp_target_maps_t = kp_map_transform_cls(ex_roi, gt_keyPoints[i], offset)
                    gt_keyPoints_inv, gt_keyPoints_inv_bg = kp_map_transform_inv_cls_bg(ex_roi, kp_target_maps_t, offset)
                    visual_attribute(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, [ex_roi], [gt_keyPoints_inv], [gt_keyPoints_inv_bg])
                    # gt_keyPoints_inv, gt_keyPoints_inv_bg = kp_map_transform_inv_reg_bg(ex_roi, kp_target_maps_t, offset)
                    # visual_attribute(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, [ex_roi], [gt_keyPoints_inv], [gt_keyPoints_inv_bg])

                    bf_targets_maps_t = kp_maps_to_bf_map(kp_target_maps_t)
                    pre_bfpoints_inv = bf_map_transform_inv_cls(ex_roi, bf_targets_maps_t, offset)
                    visual_attribute(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, [ex_roi], [pre_bfpoints_inv])

                cfg.TRAIN.MAP_SIZE = temp_size
    return kp_target_maps


def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    if cfg.TRAIN.POSITION_ROIS_NUM_ADAPT:
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        fg_rois_per_this_image = fg_inds.size
        bg_rois_per_this_image = fg_rois_per_this_image
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_rois_per_this_image = bg_rois_per_this_image if bg_rois_per_this_image < bg_inds.size else bg_inds.size
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    else:
        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds
    
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    
    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # print 'proposal_target_layer:', bbox_target_data
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights


def _sample_rois_attributes(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, gt_keyPoints, num_classes, all_rois_keyPoints=None, gt_attributes=None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    if cfg.TRAIN.POSITION_ROIS_NUM_ADAPT:
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        fg_rois_per_this_image = fg_inds.size
        bg_rois_per_this_image = fg_rois_per_this_image
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_rois_per_this_image = bg_rois_per_this_image if bg_rois_per_this_image < bg_inds.size else bg_inds.size
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    else:
        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds

    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    # TODO use offset between rois and gt_keyPoints as target to predict
    # gt_keyPoints_data = gt_keyPoints[gt_assignment[keep_inds], :]
    if cfg.TRAIN.RPN_KP_REGRESSION:
        # timer = Timer()
        # timer.tic()
        rois_keyPoints = all_rois_keyPoints[keep_inds]
        gt_keyPoints_data = _compute_keyPoints_targets(
            rois[:, 1:5], gt_keyPoints[gt_assignment[keep_inds], :], labels,
            gt_boxes[gt_assignment[keep_inds], :4], rois_keyPoints[:, 1:])
        # timer.toc()
        # print ('proposal target took {:.3f}s').format(timer.total_time)
    else:
        gt_keyPoints_data = _compute_keyPoints_targets(
            rois[:, 1:5], gt_keyPoints[gt_assignment[keep_inds], :], labels,
            gt_boxes[gt_assignment[keep_inds], :4])

    # print 'proposal_target_layer:', bbox_target_data
    bbox_targets, bbox_inside_weights, keyPoint_targets, keyPoint_inside_weights = \
        _get_bbox_regression_labels_atttribute(bbox_target_data, num_classes, gt_keyPoints_data)

    if cfg.TRAIN.KP == 6:
        attribute_targets = {}
        fg_inds_gt = gt_assignment[fg_inds]
        for attr in cfg.TRAIN.ATTRIBUTES:
            attrName = attr.keys()[0]
            if attrName != 'gt_keyPoints':
                gt_attribute = gt_attributes[attrName]
                attribute_targets[attrName] = np.zeros(len(labels))
                attribute_targets[attrName].fill(-1)
                attribute_targets[attrName][:fg_rois_per_this_image] = gt_attribute[fg_inds_gt]

    else:
        attribute_targets = None

    return labels, rois, bbox_targets, bbox_inside_weights, keyPoint_targets, keyPoint_inside_weights, attribute_targets


def _sample_rois_attributes_gen_v2_bfmap(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, gt_keyPoints, num_classes, all_rois_keyPoints=None, gt_attributes=None):
    labels, rois, bbox_targets, bbox_inside_weights, keyPoint_targets, \
    attribute_targets = _sample_rois_attributes_gen_v2(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, gt_keyPoints, num_classes, all_rois_keyPoints, gt_attributes)
    bf_targets = kp_maps_to_bf_maps(keyPoint_targets)
    return labels, rois, bbox_targets, bbox_inside_weights, keyPoint_targets, \
    attribute_targets, bf_targets


def _sample_rois_attributes_gen_v2(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, gt_keyPoints, num_classes, all_rois_keyPoints=None, gt_attributes=None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    if cfg.TRAIN.POSITION_ROIS_NUM_ADAPT:
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        fg_rois_per_this_image = fg_inds.size
        bg_rois_per_this_image = fg_rois_per_this_image
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_rois_per_this_image = bg_rois_per_this_image if bg_rois_per_this_image < bg_inds.size else bg_inds.size
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    else:
        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds

    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    # TODO use offset between rois and gt_keyPoints as target to predict

    if cfg.TRAIN.KP == 5 or cfg.TRAIN.KP == 6:
        if cfg.TRAIN.RPN_KP_REGRESSION:
            rois_keyPoints = all_rois_keyPoints[keep_inds]
            gt_keyPoints_data = _compute_keyPoints_targets(
                rois[:, 1:5], gt_keyPoints[gt_assignment[keep_inds], :], labels,
                gt_boxes[gt_assignment[keep_inds], :4], rois_keyPoints[:, 1:])
            # timer.toc()
            # print ('proposal target took {:.3f}s').format(timer.total_time)
        else:
            keyPoint_targets = _compute_keyPoints_targets_v2(
                rois[:, 1:5], gt_keyPoints[gt_assignment[keep_inds], :], labels,
                gt_boxes[gt_assignment[keep_inds], :4])

        # print 'proposal_target_layer:', bbox_target_data
        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels(bbox_target_data, num_classes)
        if cfg.TRAIN.KP == 6:
            attribute_targets = {}
            fg_inds_gt = gt_assignment[fg_inds]
            for attr in cfg.TRAIN.ATTRIBUTES:
                attrName = attr.keys()[0]
                if attrName != 'gt_keyPoints':
                    gt_attribute = gt_attributes[attrName]
                    attribute_targets[attrName] = np.zeros(len(labels))
                    attribute_targets[attrName].fill(-1)
                    attribute_targets[attrName][:fg_rois_per_this_image] = gt_attribute[fg_inds_gt]
        else:
            attribute_targets = None

    elif cfg.TRAIN.KP == 7:
        keyPoint_targets = None
        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels(bbox_target_data, num_classes)
        attribute_targets = {}
        fg_inds_gt = gt_assignment[fg_inds]
        for attr in cfg.TRAIN.ATTRIBUTES:
            attrName = attr.keys()[0]
            gt_attribute = gt_attributes[attrName]
            attribute_targets[attrName] = np.zeros(len(labels))
            attribute_targets[attrName].fill(-1)
            attribute_targets[attrName][:fg_rois_per_this_image] = gt_attribute[fg_inds_gt]


    return labels, rois, bbox_targets, bbox_inside_weights, keyPoint_targets, attribute_targets


def _sample_rois_attributes_gen(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, gt_keyPoints, num_classes, all_rois_keyPoints=None, gt_attributes=None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    if cfg.TRAIN.POSITION_ROIS_NUM_ADAPT:
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        fg_rois_per_this_image = fg_inds.size
        bg_rois_per_this_image = fg_rois_per_this_image
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_rois_per_this_image = bg_rois_per_this_image if bg_rois_per_this_image < bg_inds.size else bg_inds.size
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    else:
        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds

    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    # TODO use offset between rois and gt_keyPoints as target to predict

    if cfg.TRAIN.KP == 5 or cfg.TRAIN.KP == 6:
        if cfg.TRAIN.RPN_KP_REGRESSION:
            rois_keyPoints = all_rois_keyPoints[keep_inds]
            gt_keyPoints_data = _compute_keyPoints_targets(
                rois[:, 1:5], gt_keyPoints[gt_assignment[keep_inds], :], labels,
                gt_boxes[gt_assignment[keep_inds], :4], rois_keyPoints[:, 1:])
            # timer.toc()
            # print ('proposal target took {:.3f}s').format(timer.total_time)
        else:
            gt_keyPoints_data = _compute_keyPoints_targets(
                rois[:, 1:5], gt_keyPoints[gt_assignment[keep_inds], :], labels,
                gt_boxes[gt_assignment[keep_inds], :4])

        # print 'proposal_target_layer:', bbox_target_data
        bbox_targets, bbox_inside_weights, keyPoint_targets, keyPoint_inside_weights = \
            _get_bbox_regression_labels_atttribute(bbox_target_data, num_classes, gt_keyPoints_data)
        if cfg.TRAIN.KP == 6:
            attribute_targets = {}
            fg_inds_gt = gt_assignment[fg_inds]
            for attr in cfg.TRAIN.ATTRIBUTES:
                attrName = attr.keys()[0]
                if attrName != 'gt_keyPoints':
                    gt_attribute = gt_attributes[attrName]
                    attribute_targets[attrName] = np.zeros(len(labels))
                    attribute_targets[attrName].fill(-1)
                    attribute_targets[attrName][:fg_rois_per_this_image] = gt_attribute[fg_inds_gt]

        else:
            attribute_targets = None

    elif cfg.TRAIN.KP == 7:
        keyPoint_targets = None
        keyPoint_inside_weights = None
        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels(bbox_target_data, num_classes)
        attribute_targets = {}
        fg_inds_gt = gt_assignment[fg_inds]
        for attr in cfg.TRAIN.ATTRIBUTES:
            attrName = attr.keys()[0]
            gt_attribute = gt_attributes[attrName]
            attribute_targets[attrName] = np.zeros(len(labels))
            attribute_targets[attrName].fill(-1)
            attribute_targets[attrName][:fg_rois_per_this_image] = gt_attribute[fg_inds_gt]


    return labels, rois, bbox_targets, bbox_inside_weights, keyPoint_targets, keyPoint_inside_weights, attribute_targets


def _sample_rois_for_ohem(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):

    """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = fg_inds.size
    # Sample foreground regions without replacement

    # if fg_inds.size > 0:
    #     fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = bg_inds.size
    # Sample background regions without replacement

    # if bg_inds.size > 0:
    #     bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0  # label of pred roi: fg1 bg0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights


def _sample_rois_for_rohem(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    if cfg.TRAIN.POSITION_ROIS_NUM_ADAPT:
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        fg_rois_per_this_image = fg_inds.size
        bg_rois_per_this_image = fg_rois_per_this_image
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_rois_per_this_image = bg_rois_per_this_image if bg_rois_per_this_image < bg_inds.size else bg_inds.size
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    else:
        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.Frozen_FG_THRESH)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds

    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # print 'proposal_target_layer:', bbox_target_data
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
