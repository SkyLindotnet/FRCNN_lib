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
from fast_rcnn.bbox_transform import bbox_transform, keyPoints_transform, keyPoints_transform_inv
from utils.cython_bbox import bbox_overlaps
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from utils.timer import Timer

DEBUG = False

class kpMapOhemLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        top[0].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        kp_score = bottom[0].data
        kp_roi_gt = bottom[1].data
        kp_roi_gt_ohem = np.zeros(kp_roi_gt.shape)
        kp_roi_gt_ohem.fill(-1)
        for i in range(kp_roi_gt.shape[0]):
            kp_map = kp_roi_gt[i][0]
            kp_map_i = np.where(kp_map > 0)
            kp_roi_gt_ohem[i][0][kp_map_i] = kp_map[kp_map_i]
            fg_num = len(kp_map_i[0])
            bg_num = int(fg_num / cfg.TRAIN.KP_MAP_FG_FRACTION * (1 - cfg.TRAIN.KP_MAP_FG_FRACTION))
            bg_map_i = np.where(kp_roi_gt_ohem[i][0] == -1)
            bp_score_hard = np.argsort(kp_score[i][0][bg_map_i])[:bg_num]
            kp_roi_gt_ohem[i][0][(bg_map_i[0][bp_score_hard],
                                  bg_map_i[1][bp_score_hard])] = 0
            assert len(np.where(kp_roi_gt_ohem[i][0] > 0)[0]) == fg_num and \
                   len(np.where(kp_roi_gt_ohem[i][0] == 0)[0]) == bg_num

        top[0].reshape(*kp_roi_gt_ohem.shape)
        top[0].data[...] = kp_roi_gt_ohem

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
