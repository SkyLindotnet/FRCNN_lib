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

class DebugLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        print bottom[0].shape
        print top[0].shape

    def forward(self, bottom, top):
        print bottom[0].shape
        print top[0].shape

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
