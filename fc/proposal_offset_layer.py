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
    keyPoints_transform, keyPoints_transform_inv, kp_map_transform_v1, kp_map_transform_inv_v1, kp_map_transform_inv_v1_bg
from utils.cython_bbox import bbox_overlaps
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from utils.timer import Timer

DEBUG = False

class ProposalOffsetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5, 1, 1)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # perform offset for rois
        all_rois = all_rois.reshape(all_rois.shape[0], all_rois.shape[1])
        all_rois = all_rois[:, 1:5]
        rois = box_offset(all_rois, cfg.TRAIN.MAP_ROI_offset)  # cfg.TRAIN.MAP_ROI_offset
        # debug
        if 0:
            visual_im(cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE, rois, all_rois)

        batch_inds = np.zeros((rois.shape[0], 1), dtype=np.float32)
        rois = np.hstack((batch_inds, rois.astype(np.float32, copy=False)))

        # modified by ywxiong
        rois = rois.reshape((rois.shape[0], rois.shape[1], 1, 1))
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def visual_im(scale, gt_boxes, gt_boxes_o):
    if cfg.TRAIN.VISUAL_ANCHORS_IMG != '':
        plt.figure()
        ax = plt.subplot(1, 2, 1)
        im = cv2.imread(cfg.TRAIN.VISUAL_ANCHORS_IMG)
        if cfg.TRAIN.VISUAL_ANCHORS_IMG_Flipped:
            im = im[:, ::-1, :]

        im = cv2.resize(im, None, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_LINEAR)
        ax.imshow(im[:, :, ::-1], aspect='equal')

        for gt_boxe_o in gt_boxes_o:
            gt_boxe_o_w = gt_boxe_o[2] - gt_boxe_o[0]
            gt_boxe_o_h = gt_boxe_o[3] - gt_boxe_o[1]
            rec = Rectangle((gt_boxe_o[0], gt_boxe_o[1]), width=gt_boxe_o_w, height=gt_boxe_o_h,
                            ec='r', fill=False, lw=1.5)
            ax.add_patch(rec)

        ax = plt.subplot(1, 2, 2)
        # im = cv2.resize(im, None, None, fx=scale, fy=scale,
        #                 interpolation=cv2.INTER_LINEAR)
        ax.imshow(im[:, :, ::-1], aspect='equal')
        for gt_boxe in gt_boxes:
            gt_boxe_w = gt_boxe[2] - gt_boxe[0]
            gt_boxe_h = gt_boxe[3] - gt_boxe[1]
            rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                            ec='r', fill=False, lw=1.5)
            ax.add_patch(rec)
        plt.close('all')

def box_offset(ex_roi, offset_scale):
    ex_roi_t = ex_roi.copy()
    ex_roi_w = ex_roi[:, 2] - ex_roi[:, 0]
    ex_roi_h = ex_roi[:, 3] - ex_roi[:, 1]
    ex_roi_t_w = ex_roi_w * offset_scale
    ex_roi_t_h = ex_roi_h * offset_scale
    ex_roi_t[:, 0] = ex_roi[:, 0] - (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[:, 2] = ex_roi[:, 2] + (ex_roi_t_w - ex_roi_w) / 2
    ex_roi_t[:, 1] = ex_roi[:, 1] - (ex_roi_t_h - ex_roi_h) / 2
    ex_roi_t[:, 3] = ex_roi[:, 3] + (ex_roi_t_h - ex_roi_h) / 2
    return ex_roi_t
