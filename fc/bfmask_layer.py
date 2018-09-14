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

class BfMaskLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        top[0].reshape(1, bottom[0].data.shape[1],
                       bottom[0].data.shape[2],
                       bottom[0].data.shape[3])

    def forward(self, bottom, top):
        classNum = bottom[0].data.shape[1] - 1
        bg_prob = bottom[1].data[:, :1, :, :]
        fg_prob = bottom[1].data[:, 1:, :, :]
        fg_mask = np.tile(fg_prob, (1, classNum, 1, 1))
        bf_mask = np.hstack([bg_prob, fg_mask])
        top[0].reshape(*bf_mask.shape)
        top[0].data[...] = bf_mask

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
