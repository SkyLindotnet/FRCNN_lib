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
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes, keyPoints_transform_inv
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        if cfg.TRAIN.MULTI_SCALE_RPN == 1:
            layer_params = yaml.load(self.param_str)
            self._anchors = []
            self._num_anchors = []
            self._feat_stride = cfg.TRAIN.MULTI_SCALE_RPN_STRIDE
            for i, ANCHOR_SCALES in enumerate(cfg.TRAIN.MULTI_SCALE_RPN_SCALE):
                anchors = generate_anchors(base_size=self._feat_stride[i], scales=np.array(ANCHOR_SCALES))
                self._anchors.append(anchors)
                self._num_anchors.append(anchors.shape[0])
        else:
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

        if cfg.TRAIN.RPN_KP_REGRESSION:
            top[1].reshape(1, cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] + 1)
            # scores blob: holds scores for R regions of interest
            if len(top) > 2:
                top[2].reshape(1, 1, 1, 1)
        else:
            # scores blob: holds scores for R regions of interest
            if len(top) > 1:
                top[1].reshape(1, 1, 1, 1)

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

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str('TRAIN' if self.phase == 0 else 'TEST') # either 'TRAIN' or 'TEST'
        if cfg.TRAIN.FrozenTraing:
            cfg_key = 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE
        if cfg.TRAIN.MULTI_SCALE_RPN == 1:
            rpn_proposals = []
            rpn_scores = []
            for i, ANCHOR_SCALES in enumerate(cfg.TRAIN.MULTI_SCALE_RPN_SCALE):
                pre_nms_topN = cfg[cfg_key].MULTI_SCALE_RPN_PRE_NMS_TOP_Ns[i]
                post_nms_topN = cfg[cfg_key].MULTI_SCALE_RPN_POST_NMS_TOP_Ns[i]
                # the first set of _num_anchors channels are bg probs
                # the second set are the fg probs, which we want
                scores = bottom[0+i*2].data[:, self._num_anchors[i]:, :, :]
                bbox_deltas = bottom[1+i*2].data
                im_info = bottom[-1].data[0, :]

                # 1. Generate proposals from bbox deltas and shifted anchors
                height, width = scores.shape[-2:]

                # Enumerate all shifts
                shift_x = np.arange(0, width) * self._feat_stride[i]
                shift_y = np.arange(0, height) * self._feat_stride[i]
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                    shift_x.ravel(), shift_y.ravel())).transpose()

                # Enumerate all shifted anchors:
                #
                # add A anchors (1, A, 4) to
                # cell K shifts (K, 1, 4) to get
                # shift anchors (K, A, 4)
                # reshape to (K*A, 4) shifted anchors
                A = self._num_anchors[i]
                K = shifts.shape[0]
                anchors = self._anchors[i].reshape((1, A, 4)) + \
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
                nms_keep = nms(np.hstack((proposals, scores)), nms_thresh)
                if post_nms_topN > 0:
                    nms_keep = nms_keep[:post_nms_topN]
                proposals = proposals[nms_keep, :]
                scores = scores[nms_keep]
                rpn_proposals.append(proposals)
                rpn_scores.append(scores)

            # concat several groups of proposals from other rpn maps
            # concat all proposals
            proposals = np.vstack(rpn_proposals)
            scores = np.vstack(rpn_scores)

            # Output rois blob
            # Our RPN implementation only supports a single input image, so all
            # batch inds are 0
            batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
            blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
            # print blob.shape
            top[0].reshape(*(blob.shape))
            top[0].data[...] = blob

            if cfg.TRAIN.RPN_KP_REGRESSION:
                # timer = Timer()
                # timer.tic()
                keyPoint_deltas = bottom[3].data
                keyPoints_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
                keyPoint_deltas = keyPoint_deltas.transpose((0, 2, 3, 1)).reshape((-1, keyPoints_num))

                # m 1
                # keyPoint_proposals = keyPoints_transform_inv(anchors, keyPoint_deltas)
                # keyPoint_proposals = keyPoint_proposals[keep, :]
                # keyPoint_proposals = keyPoint_proposals[order, :]
                # keyPoint_proposals = keyPoint_proposals[nms_keep, :]

                # m2
                anchors_t = anchors[keep, :]
                anchors_t = anchors_t[order, :]
                anchors_t = anchors_t[nms_keep, :]
                keyPoint_deltas_t = keyPoint_deltas[keep, :]
                keyPoint_deltas_t = keyPoint_deltas_t[order, :]
                keyPoint_deltas_t = keyPoint_deltas_t[nms_keep, :]
                keyPoint_proposals = keyPoints_transform_inv(anchors_t, keyPoint_deltas_t)

                blob = np.hstack((batch_inds, keyPoint_proposals.astype(np.float32, copy=False)))
                # print blob.shape
                top[1].reshape(*(blob.shape))
                top[1].data[...] = blob

                # [Optional] output scores blob
                if len(top) > 2:
                    top[2].reshape(*(scores.shape))
                    top[2].data[...] = scores
                # timer.toc()
                # print ('proposal took {:.3f}s').format(timer.total_time)
            else:
                # [Optional] output scores blob
                if len(top) > 1:
                    top[1].reshape(*(scores.shape))
                    top[1].data[...] = scores
        else:
            # the first set of _num_anchors channels are bg probs
            # the second set are the fg probs, which we want
            scores = bottom[0].data[:, self._num_anchors:, :, :]
            bbox_deltas = bottom[1].data
            im_info = bottom[2].data[0, :]

            if DEBUG:
                print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                print 'scale: {}'.format(im_info[2])

            # 1. Generate proposals from bbox deltas and shifted anchors
            height, width = scores.shape[-2:]

            if DEBUG:
                print 'score map size: {}'.format(scores.shape)

            # Enumerate all shifts
            shift_x = np.arange(0, width) * self._feat_stride
            shift_y = np.arange(0, height) * self._feat_stride
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

            if len(bottom) > 30:
                # forward anchor from con4_3 and combine all anchor

                # the first set of _num_anchors channels are bg probs
                # the second set are the fg probs, which we want
                scores_from_conv4_3 = bottom[3].data[:, self._num_anchors_from_conv4_3:, :, :]
                bbox_deltas_from_conv4_3 = bottom[4].data
                im_info = bottom[2].data[0, :]

                if DEBUG:
                    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                    print 'scale: {}'.format(im_info[2])

                # 1. Generate proposals from bbox deltas and shifted anchors
                height_from_conv4_3, width_from_conv4_3 = scores_from_conv4_3.shape[-2:]

                if DEBUG:
                    print 'score map size: {}'.format(scores_from_conv4_3.shape)

                # Enumerate all shifts
                shift_x = np.arange(0, width_from_conv4_3) * self._feat_stride_from_conv4_3
                shift_y = np.arange(0, height_from_conv4_3) * self._feat_stride_from_conv4_3
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                shifts_from_conv4_3 = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                    shift_x.ravel(), shift_y.ravel())).transpose()

                # Enumerate all shifted anchors:
                #
                # add A anchors (1, A, 4) to
                # cell K shifts (K, 1, 4) to get
                # shift anchors (K, A, 4)
                # reshape to (K*A, 4) shifted anchors
                A = self._num_anchors_from_conv4_3
                K = shifts_from_conv4_3.shape[0]
                anchors_from_conv4_3 = self._anchors_from_conv4_3.reshape((1, A, 4)) + \
                          shifts_from_conv4_3.reshape((1, K, 4)).transpose((1, 0, 2))
                anchors_from_conv4_3 = anchors_from_conv4_3.reshape((K * A, 4))

                # Transpose and reshape predicted bbox transformations to get them
                # into the same order as the anchors:
                #
                # bbox deltas will be (1, 4 * A, H, W) format
                # transpose to (1, H, W, 4 * A)
                # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
                # in slowest to fastest order
                bbox_deltas_from_conv4_3 = bbox_deltas_from_conv4_3.transpose((0, 2, 3, 1)).reshape((-1, 4))

                # Same story for the scores:
                #
                # scores are (1, A, H, W) format
                # transpose to (1, H, W, A)
                # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
                scores_from_conv4_3 = scores_from_conv4_3.transpose((0, 2, 3, 1)).reshape((-1, 1))

                # Convert anchors into proposals via bbox transformations
                proposals_from_conv4_3 = bbox_transform_inv(anchors_from_conv4_3, bbox_deltas_from_conv4_3)

                # 2. clip predicted boxes to image
                proposals_from_conv4_3 = clip_boxes(proposals_from_conv4_3, im_info[:2])

                # 3. remove predicted boxes with either height or width < threshold
                # (NOTE: convert min_size to input image scale stored in im_info[2])
                keep = _filter_boxes(proposals_from_conv4_3, min_size * im_info[2])
                proposals_from_conv4_3 = proposals_from_conv4_3[keep, :]
                scores_from_conv4_3 = scores_from_conv4_3[keep]

                # 3.1 combine all anchor from conv4_3 and conv5_3
                # strategy 1
                # proposals = np.vstack((proposals, proposals_from_conv4_3))
                # scores = np.vstack((scores, scores_from_conv4_3))

                # strategy 2
                # sort all anchors from conv5_3 and conv4_3 and use nmx before combine them
                # order = scores.ravel().argsort()[::-1]
                # if pre_nms_topN > 0:
                #     order = order[:pre_nms_topN]
                # proposals = proposals[order, :]
                # scores = scores[order]
                # keep = nms(np.hstack((proposals, scores)), nms_thresh)
                # if post_nms_topN > 0:
                #     keep = keep[:post_nms_topN]
                # proposals = proposals[keep, :]
                # scores = scores[keep]
                #
                # order = scores_from_conv4_3.ravel().argsort()[::-1]
                # if pre_nms_topN > 0:
                #     order = order[:pre_nms_topN]
                # proposals_from_conv4_3 = proposals_from_conv4_3[order, :]
                # scores_from_conv4_3 = scores_from_conv4_3[order]
                # keep = nms(np.hstack((proposals_from_conv4_3, scores_from_conv4_3)), nms_thresh)
                # if post_nms_topN > 0:
                #     keep = keep[:post_nms_topN]
                # proposals_from_conv4_3 = proposals_from_conv4_3[keep, :]
                # scores_from_conv4_3 = scores_from_conv4_3[keep]
                #
                # proposals = np.vstack((proposals, proposals_from_conv4_3))
                # scores = np.vstack((scores, scores_from_conv4_3))

                # strategy 3
                # proposals = proposals_from_conv4_3
                # scores = scores_from_conv4_3
                # ------------------------------

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
            nms_keep = nms(np.hstack((proposals, scores)), nms_thresh)
            if post_nms_topN > 0:
                nms_keep = nms_keep[:post_nms_topN]
            proposals = proposals[nms_keep, :]
            scores = scores[nms_keep]

            # if cfg.RPN_FILTER:
            #     scores_i = np.where(scores[:, 0] > cfg.RPN_FILTER_thresh)
            #     if len(scores_i[0]) == 0:
            #         proposals = proposals[:5]
            #         scores = scores[:5]
            #     else:
            #         proposals = proposals[scores_i]
            #         scores = scores[scores_i]
            #     areas = (proposals[:,2]-proposals[:,0])*(proposals[:,3]-proposals[:,1])
            #     argmax_area_i = np.argmax(areas)
            #     proposals = proposals[argmax_area_i, np.newaxis]
            #     scores = scores[argmax_area_i, np.newaxis]

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

            if cfg.TRAIN.RPN_KP_REGRESSION:
                # timer = Timer()
                # timer.tic()
                keyPoint_deltas = bottom[3].data
                keyPoints_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
                keyPoint_deltas = keyPoint_deltas.transpose((0, 2, 3, 1)).reshape((-1, keyPoints_num))

                # m 1
                # keyPoint_proposals = keyPoints_transform_inv(anchors, keyPoint_deltas)
                # keyPoint_proposals = keyPoint_proposals[keep, :]
                # keyPoint_proposals = keyPoint_proposals[order, :]
                # keyPoint_proposals = keyPoint_proposals[nms_keep, :]

                # m2
                anchors_t = anchors[keep, :]
                anchors_t = anchors_t[order, :]
                anchors_t = anchors_t[nms_keep, :]
                keyPoint_deltas_t = keyPoint_deltas[keep, :]
                keyPoint_deltas_t = keyPoint_deltas_t[order, :]
                keyPoint_deltas_t = keyPoint_deltas_t[nms_keep, :]
                keyPoint_proposals = keyPoints_transform_inv(anchors_t, keyPoint_deltas_t)

                blob = np.hstack((batch_inds, keyPoint_proposals.astype(np.float32, copy=False)))
                # print blob.shape
                top[1].reshape(*(blob.shape))
                top[1].data[...] = blob

                # [Optional] output scores blob
                if len(top) > 2:
                    top[2].reshape(*(scores.shape))
                    top[2].data[...] = scores
                # timer.toc()
                # print ('proposal took {:.3f}s').format(timer.total_time)
            else:
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
