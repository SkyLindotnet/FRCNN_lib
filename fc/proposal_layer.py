# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from rpn.generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes, keyPoints_transform_inv
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from roi_data_layer.minibatch import transform_kp_to_box

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML

        # layer_params = yaml.load(self.param_str)
        # self._feat_stride = layer_params['feat_stride']
        # # anchor_scales = layer_params.get('scales', (8, 16, 32))
        # if cfg.PYRAMID_MORE:
        #     DEFALU_ANCHOR_SCALES = cfg.PYRAMID_MORE_ANCHORS[-1]
        # else:
        #     DEFALU_ANCHOR_SCALES = cfg.DEFALU_ANCHOR_SCALES
        # anchor_scales = layer_params.get('scales', DEFALU_ANCHOR_SCALES)  # 8, 16, 32
        #
        # self._anchors = generate_anchors(scales=np.array(anchor_scales))
        # self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        if cfg.TRAIN.RPN_KP_REGRESSION:
            top[1].reshape(1, cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] + 1)
            # scores blob: holds scores for R regions of interest
            if len(top) > 2:
                top[2].reshape(1, 1, 1, 1)
        else:
            # scores blob: holds scores for R regions of interest
            if len(top) > 1:
                top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        cfg_key = str('TRAIN' if self.phase == 0 else 'TEST')  # either 'TRAIN' or 'TEST'
        # cfg_key = 'TRAIN'
        pre_nms_topN  = cfg[cfg_key].Frozen_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].Frozen_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].Frozen_NMS_THRESH
        min_size      = cfg[cfg_key].Frozen_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, 1].reshape(-1, 1)
        bbox_deltas = bottom[1].data[:, 4:]
        im_info = bottom[2].data[0, :]
        rois = bottom[3].data[:, 1:5]

        # reshape rois (-1, 4)
        if len(rois.shape) == 4:
            rois = rois.reshape(rois.shape[0], rois.shape[1])

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(rois, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if cfg[cfg_key].Frozen_NMS:
            nms_keep = nms(np.hstack((proposals, scores)), nms_thresh)
            if post_nms_topN > 0:
                nms_keep = nms_keep[:post_nms_topN]
            proposals = proposals[nms_keep, :]
            scores = scores[nms_keep]

        # concat several groups of proposals from other rpn maps

        # using gt as roi
        if cfg[cfg_key].USING_GT:
            gt_kps = cfg.TRAIN.ANNOINFOS[:, 5:]
            gt_boxes = cfg.TRAIN.ANNOINFOS[:, :4]
            if cfg.TRANSFORM_KP_TO_BOX:
                gt_boxes = transform_kp_to_box(gt_kps, gt_boxes, cfg.TRAIN.VISUAL_ANCHORS_IMG)
            proposals = gt_boxes * cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE
            scores = cfg.TRAIN.ANNOINFOS[:, 4].reshape([1, 1])

        # replace undected face with image coordinate
        if cfg.KP_UNDETECTED_FILL and scores.max() < cfg.KP_UNDETECTED_FILL_THRESH:
            offset_x = (im_info[1] - im_info[1] / cfg.TRAIN.MAP_offset) / 2
            offset_y = (im_info[0] - im_info[0] / cfg.TRAIN.MAP_offset) / 2
            # proposals = np.array([[0, 0, im_info[1], im_info[0]]])
            proposals = np.array([[offset_x, offset_y, im_info[1]-offset_x, im_info[0]-offset_y]])
            scores = np.array([[1]])

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        # print blob.shape
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
