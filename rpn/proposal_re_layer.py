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
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = False

class ProposalRELayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._feat_stride = layer_params['feat_stride']
        # anchor_scales = layer_params.get('scales', (8, 16, 32))
        if cfg.PYRAMID_MORE:
            DEFALU_ANCHOR_SCALES = cfg.PYRAMID_MORE_ANCHORS[-1]
        else:
            DEFALU_ANCHOR_SCALES = cfg.DEFALU_ANCHOR_SCALES
        anchor_scales = layer_params.get('scales', DEFALU_ANCHOR_SCALES)  # 8, 16, 32

        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

        if cfg.PYRAMID_MORE:
            assert len(cfg.PYRAMID_MORE_ANCHORS) == len(layer_params)
            self._anchors_from_extends = []
            self._num_anchors_from_extends = []
            self._feat_stride_from_extends = []
            PYRAMID_NUM = len(layer_params)
            for i, j in zip(range(4, 5-PYRAMID_NUM, -1), range(1, PYRAMID_NUM)):
                feat_stride_str = "feat_stride_from_p%d_3" % i
                anchor_scales_from_extend = layer_params.get('scales', cfg.PYRAMID_MORE_ANCHORS[PYRAMID_NUM-1-j])
                # generate anchors with different raido and scale
                self._anchors_from_extend = generate_anchors(base_size=layer_params[feat_stride_str],
                                                             scales=np.array(anchor_scales_from_extend))
                self._num_anchors_from_extend = self._anchors_from_extend.shape[0]
                self._feat_stride_from_extend = layer_params[feat_stride_str]
                # record info of anchors from extend convs
                self._anchors_from_extends.append(self._anchors_from_extend)
                self._num_anchors_from_extends.append(self._num_anchors_from_extend)
                self._feat_stride_from_extends.append(self._feat_stride_from_extend)
                top[j].reshape(1, 5)


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

        # assert bottom[0].data.shape[0] == 1, \
        #     'Only single item batches are supported'

        cfg_key = str('TRAIN' if self.phase == 0 else 'TEST') # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RE_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RE_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RE_NMS_THRESH
        min_size      = cfg[cfg_key].RE_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, 1].reshape(-1, 1)  # (num ,1)
        bbox_deltas = bottom[1].data[:, 4:]  # (num ,4)
        im_info = bottom[2].data[0, :]
        rois = bottom[3].data[:, 1:5]

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals_re from bbox deltas and shifted anchors
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
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # concat several groups of proposals from other rpn maps
        if cfg.RPN_PYRAMID_MORE:
            RPN_PYRAMID_NUM = cfg.RPN_PYRAMID_NUM
            for j in range(1, RPN_PYRAMID_NUM):
                # the first set of _num_anchors channels are bg probs
                # the second set are the fg probs, which we want
                scores_extend = bottom[1+2*j].data[:, self._num_anchors:, :, :]
                bbox_deltas = bottom[2+2*j].data

                if DEBUG:
                    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                    print 'scale: {}'.format(im_info[2])

                # 1. Generate proposals from bbox deltas and shifted anchors
                height_extend, width_extend = scores_extend.shape[-2:]

                if DEBUG:
                    print 'score map size: {}'.format(scores_extend.shape)

                # Enumerate all shifts
                shift_x = np.arange(0, width_extend) * self._feat_stride
                shift_y = np.arange(0, height_extend) * self._feat_stride
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                    shift_x.ravel(), shift_y.ravel())).transpose()

                # Enumerate all shifted anchors:
                #
                # add A anchors (1, A, 4) to
                # cell K shifts (K, 1, 4) to get
                # shift anchors (K, A, 4)
                # reshape to (K*A, 4) shifted anchors
                A = self._num_anchors
                K = shifts.shape[0]
                anchors = self._anchors.reshape((1, A, 4)) + \
                          shifts.reshape((1, K, 4)).transpose((1, 0, 2))
                anchors = anchors.reshape((K * A, 4))

                # Transpose and reshape predicted bbox transformations to get them
                # into the same order as the anchors:
                #
                # bbox deltas will be (1, 4 * A, H, W) format
                # transpose to (1, H, W, 4 * A)
                # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
                # in slowest to fastest order
                bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

                # Same story for the scores:
                #
                # scores are (1, A, H, W) format
                # transpose to (1, H, W, A)
                # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
                scores_extend = scores_extend.transpose((0, 2, 3, 1)).reshape((-1, 1))

                # Convert anchors into proposals via bbox transformations
                proposals_extend = bbox_transform_inv(anchors, bbox_deltas)

                # 2. clip predicted boxes to image
                proposals_extend = clip_boxes(proposals_extend, im_info[:2])

                # 3. remove predicted boxes with either height or width < threshold
                # (NOTE: convert min_size to input image scale stored in im_info[2])
                keep = _filter_boxes(proposals_extend, min_size * im_info[2])
                proposals_extend = proposals_extend[keep, :]
                scores_extend = scores_extend[keep]

                # 4. sort all (proposal, score) pairs by score from highest to lowest
                # 5. take top pre_nms_topN (e.g. 6000)
                order = scores_extend.ravel().argsort()[::-1]
                if pre_nms_topN > 0:
                    order = order[:pre_nms_topN]
                proposals_extend = proposals_extend[order, :]
                scores_extend = scores_extend[order]

                # 6. apply nms (e.g. threshold = 0.7)
                # 7. take after_nms_topN (e.g. 300)
                # 8. return the top proposals (-> RoIs top)
                keep = nms(np.hstack((proposals_extend, scores_extend)), nms_thresh)
                if post_nms_topN > 0:
                    keep = keep[:post_nms_topN]
                proposals_extend = proposals_extend[keep, :]
                scores_extend = scores_extend[keep]

                # 9 concat all proposals
                proposals = np.vstack((proposals, proposals_extend))
                scores = np.vstack((scores, scores_extend))

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

        if cfg.PYRAMID_MORE:
            PYRAMID_NUM = len(cfg.PYRAMID_MORE_ANCHORS)
            for i in range(1, PYRAMID_NUM):
                # the first set of _num_anchors channels are bg probs
                # the second set are the fg probs, which we want
                scores = bottom[2+2*i-1].data[:, self._num_anchors_from_extends[i-1]:, :, :]
                bbox_deltas = bottom[2+2*i].data

                # 1. Generate proposals from bbox deltas and shifted anchors
                height, width = scores.shape[-2:]

                if DEBUG:
                    print 'score map size: {}'.format(scores.shape)

                # Enumerate all shifts
                shift_x = np.arange(0, width) * self._feat_stride_from_extends[i-1]
                shift_y = np.arange(0, height) * self._feat_stride_from_extends[i-1]
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                    shift_x.ravel(), shift_y.ravel())).transpose()

                # Enumerate all shifted anchors:
                #
                # add A anchors (1, A, 4) to
                # cell K shifts (K, 1, 4) to get
                # shift anchors (K, A, 4)
                # reshape to (K*A, 4) shifted anchors
                A = self._num_anchors_from_extends[i-1]
                K = shifts.shape[0]
                anchors = self._anchors_from_extends[i-1].reshape((1, A, 4)) + \
                          shifts.reshape((1, K, 4)).transpose((1, 0, 2))
                anchors = anchors.reshape((K * A, 4))

                # Transpose and reshape predicted bbox transformations to get them
                # into the same order as the anchors:
                #
                # bbox deltas will be (1, 4 * A, H, W) format
                # transpose to (1, H, W, 4 * A)
                # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
                # in slowest to fastest order
                bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

                # Same story for the scores:
                #
                # scores are (1, A, H, W) format
                # transpose to (1, H, W, A)
                # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
                scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

                # Convert anchors into proposals via bbox transformations
                proposals = bbox_transform_inv(anchors, bbox_deltas)

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
                keep = nms(np.hstack((proposals, scores)), nms_thresh)
                if post_nms_topN > 0:
                    keep = keep[:post_nms_topN]
                proposals = proposals[keep, :]
                scores = scores[keep]

                # Output rois blob
                # Our RPN implementation only supports a single input image, so all
                # batch inds are 0
                batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
                blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
                # print blob.shape
                top[i].reshape(*(blob.shape))
                top[i].data[...] = blob


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