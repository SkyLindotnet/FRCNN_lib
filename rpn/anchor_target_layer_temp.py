# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math, cv2
from utils.timer import Timer

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def debugAnchorPosition(self, shifts, im_info, gt_boxes, valid_regions, height, width):
        if cfg.TRAIN.VISUAL_ANCHORS_IMG != '':
            plt.figure(0)
            ax = plt.subplot(1, 2, 1)
            im = cv2.imread(cfg.TRAIN.VISUAL_ANCHORS_IMG)
            if cfg.TRAIN.VISUAL_ANCHORS_IMG_Flipped:
                im = im[:, ::-1, :]
                cfg.TRAIN.VISUAL_ANCHORS_IMG_Flipped = False
            ax.imshow(im[:, :, ::-1], aspect='equal')
            ax = plt.subplot(1, 2, 2)
            im = cv2.resize(im, None, None, fx=im_info[2], fy=im_info[2],
                    interpolation=cv2.INTER_LINEAR)
            ax.imshow(im[:, :, ::-1], aspect='equal')
            for gt_boxe in gt_boxes:
                gt_boxe_w = gt_boxe[2] - gt_boxe[0]
                gt_boxe_h = gt_boxe[3] - gt_boxe[1]
                rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                                ec='r', fill=False, lw=1.5)
                ax.add_patch(rec)

        anchor_scales_v1 = (8, 16, 32)
        anchor_scales_v2 = (4, 8, 16, 32)
        anchor_scales_v3 = (2, 4, 8, 16, 32)
        anchor_scales_v4 = (1, 2, 4, 8, 16, 32)
        anchor_scales_list = [anchor_scales_v1, anchor_scales_v2,
                              anchor_scales_v3, anchor_scales_v4]

        K = shifts.shape[0]  # number of proposals is width*height
        plt.figure(1)
        for i, anchor_scales in enumerate(anchor_scales_list):
            _anchors = generate_anchors(scales=np.array(anchor_scales))
            num_anchors = _anchors.shape[0]

            all_anchors = (_anchors.reshape((1, num_anchors, 4)) +
                              shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
            all_anchors = all_anchors.reshape((K * num_anchors, 4))
            total_anchors = int(K * num_anchors)
            # only keep anchors inside the image
            inds_inside = np.where(
                (all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
                (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
            )[0]
            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]

            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]
            # select repetitive gt argmax overlaps
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            # # fg label: for each gt, anchor with highest overlap
            # labels[gt_argmax_overlaps] = 1
            #
            # # fg label: above threshold IOU
            # labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            # visual target anchor and gt
            # plt.figure(i)
            ax = plt.subplot(2, 2, i+1)
            h = im_info[0]
            w = im_info[1]
            plt.xlim(xmax=w, xmin=0)
            plt.ylim(ymax=0, ymin=h)

            for i in range(16, w, 16):
                ax.plot([i, i], [0, 1000], 'm', linestyle='dotted', lw=0.5)
            for j in range(16, h, 16):
                ax.plot([0, 1000], [j, j], 'm', linestyle='dotted', lw=0.5)
            scaleList = np.array([1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['g', 'c', 'm', 'y', 'k', 'b']
            target_anchors = anchors[np.where(max_overlaps > cfg.TRAIN.RPN_POSITIVE_OVERLAP)]
            target_anchors = np.vstack((anchors[gt_argmax_overlaps], target_anchors))
            for anchor in target_anchors:
                anchor_w = anchor[2] - anchor[0]
                anchor_h = anchor[3] - anchor[1]
                scale = int(round(math.sqrt(anchor_w*anchor_h/256)))
                index = np.where(scaleList == scale)[0]
                if len(index) == 0:
                    index = np.where(scaleList == (scale+1))[0]
                if len(index) == 0:
                    index = np.where(scaleList == (scale-1))[0]
                print 'anchor_w %s anchor_h %s scale %d' % (str(anchor_w), str(anchor_h), scale)
                assert len(index) != 0
                rec = Rectangle((anchor[0], anchor[1]), width=anchor_w, height=anchor_h,
                                ec=colors[index[0]], fill=False)
                ax.add_patch(rec)
            for gt_boxe in gt_boxes:
                gt_boxe_w = gt_boxe[2] - gt_boxe[0]
                gt_boxe_h = gt_boxe[3] - gt_boxe[1]
                rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                                ec='r', fill=False, lw=1.5)
                ax.add_patch(rec)
            # print 'num of target anchor(>0.5): %s' % target_anchors.shape[0]
            plt.title(str(anchor_scales)+'\nnum of target anchor(>0.5): %s' % target_anchors.shape[0], fontsize=12)
        plt.tight_layout()

        # add auxiliary anchors

        plt.figure(2)
        for i, anchor_scales in enumerate(anchor_scales_list):
            _anchors = generate_anchors(scales=np.array(anchor_scales))
            num_anchors = _anchors.shape[0]

            all_anchors = (_anchors.reshape((1, num_anchors, 4)) +
                              shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
            all_anchors = all_anchors.reshape((K * num_anchors, 4))
            total_anchors = int(K * num_anchors)

            # only keep anchors inside the image
            inds_inside = np.where(
                (all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
                (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
            )[0]
            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]

            # keep only valid anchors
            valid_inds = []
            all_anchors = all_anchors.reshape((height, width, num_anchors, 4))
            valid_anchors = all_anchors[valid_regions]
            valid_anchors = valid_anchors.reshape((valid_anchors.shape[0]*valid_anchors.shape[1], 4))
            for valid_anchor in valid_anchors:
                valid_inds = np.hstack([valid_inds, np.where((anchors == valid_anchor).all(1))[0]]).astype(np.int)

            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))

            # keep top n the closest anchors
            target_inds = []
            top_num = 5
            for k in range(gt_boxes.shape[0]):
                valid_overlaps = overlaps[valid_inds, k]  # ][:
                sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])

            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]
            # select repetitive gt argmax overlaps
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            # # fg label: for each gt, anchor with highest overlap
            # labels[gt_argmax_overlaps] = 1
            #
            # # fg label: above threshold IOU
            # labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            # visual target anchor and gt
            # plt.figure(i)
            ax = plt.subplot(2, 2, i+1)
            h = im_info[0]
            w = im_info[1]
            plt.xlim(xmax=w, xmin=0)
            plt.ylim(ymax=0, ymin=h)

            for i in range(16, w, 16):
                ax.plot([i, i], [0, 1000], 'm', linestyle='dotted', lw=0.5)
            for j in range(16, h, 16):
                ax.plot([0, 1000], [j, j], 'm', linestyle='dotted', lw=0.5)
            scaleList = np.array([1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['g', 'c', 'm', 'y', 'k', 'b']
            # target_anchors = anchors[np.where(max_overlaps > cfg.TRAIN.RPN_POSITIVE_OVERLAP)]
            # target_anchors = np.vstack((anchors[gt_argmax_overlaps], target_anchors))
            target_anchors = anchors[target_inds]  # valid_inds
            for anchor in target_anchors:
                anchor_w = anchor[2] - anchor[0]
                anchor_h = anchor[3] - anchor[1]
                scale = int(round(math.sqrt(anchor_w*anchor_h/256)))
                index = np.where(scaleList == scale)[0]
                if len(index) == 0:
                    index = np.where(scaleList == (scale+1))[0]
                if len(index) == 0:
                    index = np.where(scaleList == (scale-1))[0]
                print 'anchor_w %s anchor_h %s scale %d' % (str(anchor_w), str(anchor_h), scale)
                assert len(index) != 0
                rec = Rectangle((anchor[0], anchor[1]), width=anchor_w, height=anchor_h,
                                ec=colors[index[0]], fill=False)
                ax.add_patch(rec)
            for gt_boxe in gt_boxes:
                gt_boxe_w = gt_boxe[2] - gt_boxe[0]
                gt_boxe_h = gt_boxe[3] - gt_boxe[1]
                rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                                ec='r', fill=False, lw=1.5)
                ax.add_patch(rec)
            # print 'num of target anchor(>0.5): %s' % target_anchors.shape[0]
            plt.title(str(anchor_scales)+'\nnum of target anchor(>0.5): %s' % target_anchors.shape[0], fontsize=12)
        plt.tight_layout()

        # print 'done'
        plt.close('all')

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        # anchor_scales = layer_params.get('scales', (8, 16, 32))

        DEFALU_ANCHOR_SCALES = cfg.DEFALU_ANCHOR_SCALES
        anchor_scales = layer_params.get('scales', DEFALU_ANCHOR_SCALES)  # 8, 16, 32

        self._anchors = generate_anchors(scales=np.array(anchor_scales))  # generate 9 anchors with different raido and scale
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)

        if len(layer_params) > 1:
            # setup new anchor from conv4_3

            anchor_scales_from_conv4_3 = layer_params.get('scales', (2, 4, 8))  # 1 2 4; 0.6, 0.8, 1; 2, 4, 8; 4, 8, 16
            self._anchors_from_conv4_3 = generate_anchors(base_size=8, scales=np.array(anchor_scales_from_conv4_3))  # generate 9 anchors with different raido and scale
            self._num_anchors_from_conv4_3 = self._anchors_from_conv4_3.shape[0]
            self._feat_stride_from_conv4_3 = layer_params['feat_stride_from_conv4_3']

            if DEBUG:
                print 'anchors:'
                print self._anchors_from_conv4_3
                print 'anchor shapes:'
                print np.hstack((
                    self._anchors_from_conv4_3[:, 2::4] - self._anchors_from_conv4_3[:, 0::4],
                    self._anchors_from_conv4_3[:, 3::4] - self._anchors_from_conv4_3[:, 1::4],
                ))
                self._counts_from_conv4_3 = cfg.EPS
                self._sums_from_conv4_3 = np.zeros((1, 4))
                self._squared_sums_from_conv4_3 = np.zeros((1, 4))
                self._fg_sum_from_conv4_3 = 0
                self._bg_sum_from_conv4_3 = 0
                self._count_from_conv4_3 = 0

            # allow boxes to sit over the edge by a small amount
            self._allowed_border_from_conv4_3 = layer_params.get('allowed_border', 0)

            height_from_conv4_3, width_from_conv4_3 = bottom[4].data.shape[-2:]
            if DEBUG:
                print 'AnchorTargetLayer: height', height_from_conv4_3, 'width', width_from_conv4_3

            A_from_conv4_3 = self._num_anchors_from_conv4_3
            # labels_from_conv4_3
            top[4].reshape(1, 1, A_from_conv4_3 * height_from_conv4_3, width_from_conv4_3)
            # bbox_targets_from_conv4_3
            top[5].reshape(1, A_from_conv4_3 * 4, height_from_conv4_3, width_from_conv4_3)
            # bbox_inside_weights_from_conv4_3
            top[6].reshape(1, A_from_conv4_3 * 4, height_from_conv4_3, width_from_conv4_3)
            # bbox_outside_weights_from_conv4_3
            top[7].reshape(1, A_from_conv4_3 * 4, height_from_conv4_3, width_from_conv4_3)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'
        # assert bottom[4].data.shape[0] == 1, \
        #     'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]  # number of proposals is width*height
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])

        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))

        if cfg.TRAIN.EXTEND_ANCHORS_SELECTING:
            timer = Timer()
            timer.tic()
            # return valid regions according to gt_boxes
            valid_regions = _compute_valid_regions(height, width, gt_boxes, 0)
            timer.toc()
            print ('_compute_valid_regions took {:.3f}s').format(timer.total_time)
            # keep only valid anchors
            valid_inds = []
            all_anchors = all_anchors.reshape((height, width, A, 4))
            valid_anchors = all_anchors[valid_regions]
            valid_anchors = valid_anchors.reshape((valid_anchors.shape[0]*valid_anchors.shape[1], 4))

            timer = Timer()
            timer.tic()
            for valid_anchor in valid_anchors:
                valid_inds = np.hstack([valid_inds, np.where((anchors == valid_anchor).all(1))[0]]).astype(np.int)
            timer.toc()
            print ('valid_anchors hstack took {:.3f}s').format(timer.total_time)

            # keep top n the closest anchors
            target_inds = []
            top_num = 5

            timer = Timer()
            timer.tic()
            for k in range(gt_boxes.shape[0]):
                valid_overlaps = overlaps[valid_inds, k]
                sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])
            timer.toc()
            print ('gt_boxes sort took {:.3f}s').format(timer.total_time)

        # compute max overlap between every inside anchors and all gt
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # visual target anchor and gt
        if cfg.TRAIN.VISUAL_ANCHORS:
            self.debugAnchorPosition(shifts, im_info, gt_boxes, valid_regions, height, width)

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        if cfg.TRAIN.EXTEND_ANCHORS_SELECTING:
            labels[target_inds] = 1
        else:
            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps] = 1

            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            # print "was %s inds, disabling %s, now %s inds" % \
            #       (len(bg_inds), len(disable_inds), np.sum(labels == 0))

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

        if len(bottom) > 4:
            # forward anchor layer based on conv4_3

            height, width = bottom[4].data.shape[-2:]

            # 1. Generate proposals from bbox deltas and shifted anchors
            shift_x = np.arange(0, width) * self._feat_stride_from_conv4_3
            shift_y = np.arange(0, height) * self._feat_stride_from_conv4_3
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose()
            # add A anchors (1, A, 4) to
            # cell K shifts (K, 1, 4) to get
            # shift anchors (K, A, 4)
            # reshape to (K*A, 4) shifted anchors
            A = self._num_anchors_from_conv4_3
            K = shifts.shape[0]
            all_anchors = (self._anchors_from_conv4_3.reshape((1, A, 4)) +
                           shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
            all_anchors = all_anchors.reshape((K * A, 4))
            total_anchors = int(K * A)

            # only keep anchors inside the image
            inds_inside = np.where(
                (all_anchors[:, 0] >= -self._allowed_border_from_conv4_3) &
                (all_anchors[:, 1] >= -self._allowed_border_from_conv4_3) &
                (all_anchors[:, 2] < im_info[1] + self._allowed_border_from_conv4_3) &  # width
                (all_anchors[:, 3] < im_info[0] + self._allowed_border_from_conv4_3)    # height
            )[0]

            if DEBUG:
                print 'total_anchors', total_anchors
                print 'inds_inside', len(inds_inside)

            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]
            if DEBUG:
                print 'anchors.shape', anchors.shape

            # label: 1 is positive, 0 is negative, -1 is dont care
            labels = np.empty((len(inds_inside), ), dtype=np.float32)
            labels.fill(-1)

            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            # gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps] = 1

            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            # subsample positive labels if we have too many
            num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
            fg_inds = np.where(labels == 1)[0]
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(
                    fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1

            # subsample negative labels if we have too many
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
            bg_inds = np.where(labels == 0)[0]
            if len(bg_inds) > num_bg:
                disable_inds = npr.choice(
                    bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                labels[disable_inds] = -1
                #print "was %s inds, disabling %s, now %s inds" % (
                    #len(bg_inds), len(disable_inds), np.sum(labels == 0))

            bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

            bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

            bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                # uniform weighting of examples (given non-uniform sampling)
                num_examples = np.sum(labels >= 0)
                positive_weights = np.ones((1, 4)) * 1.0 / num_examples
                negative_weights = np.ones((1, 4)) * 1.0 / num_examples
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
                positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                    np.sum(labels == 1))
                negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                    np.sum(labels == 0))
            bbox_outside_weights[labels == 1, :] = positive_weights
            bbox_outside_weights[labels == 0, :] = negative_weights

            if DEBUG:
                self._sums_from_conv4_3 += bbox_targets[labels == 1, :].sum(axis=0)
                self._squared_sums_from_conv4_3 += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
                self._counts_from_conv4_3 += np.sum(labels == 1)
                means = self._sums_from_conv4_3 / self._counts_from_conv4_3
                stds = np.sqrt(self._squared_sums_from_conv4_3 / self._counts_from_conv4_3 - means ** 2)
                print 'means:'
                print means
                print 'stdevs:'
                print stds

            # map up to original set of anchors
            labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
            bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
            bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
            bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

            if DEBUG:
                print 'rpn: max max_overlap', np.max(max_overlaps)
                print 'rpn: num_positive', np.sum(labels == 1)
                print 'rpn: num_negative', np.sum(labels == 0)
                self._fg_sum_from_conv4_3 += np.sum(labels == 1)
                self._bg_sum_from_conv4_3 += np.sum(labels == 0)
                self._count_from_conv4_3 += 1
                print 'rpn: num_positive avg', self._fg_sum_from_conv4_3 / self._count_from_conv4_3
                print 'rpn: num_negative avg', self._bg_sum_from_conv4_3 / self._count_from_conv4_3

            # labels_from_conv4_3
            labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
            labels = labels.reshape((1, 1, A * height, width))
            top[4].reshape(*labels.shape)
            top[4].data[...] = labels

            # bbox_targets_from_conv4_3
            bbox_targets = bbox_targets \
                .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
            top[5].reshape(*bbox_targets.shape)
            top[5].data[...] = bbox_targets

            # bbox_inside_weights_from_conv4_3
            bbox_inside_weights = bbox_inside_weights \
                .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
            assert bbox_inside_weights.shape[2] == height
            assert bbox_inside_weights.shape[3] == width
            top[6].reshape(*bbox_inside_weights.shape)
            top[6].data[...] = bbox_inside_weights

            # bbox_outside_weights_from_conv4_3
            bbox_outside_weights = bbox_outside_weights \
                .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
            assert bbox_outside_weights.shape[2] == height
            assert bbox_outside_weights.shape[3] == width
            top[7].reshape(*bbox_outside_weights.shape)
            top[7].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
    if cfg.TRAIN.RPN_NORMALIZE_TARGETS:
        assert cfg.TRAIN.RPN_NORMALIZE_MEANS is not None
        assert cfg.TRAIN.RPN_NORMALIZE_STDS is not None
        targets -= cfg.TRAIN.RPN_NORMALIZE_MEANS
        targets /= cfg.TRAIN.RPN_NORMALIZE_STDS
    return targets

def _compute_valid_regions(height, width, gt_boxes, visual=0):
    valid_regions = [[], []]
    if visual:
        plt.figure(3)
        ax = plt.subplot(1, 1, 1)
        plt.xlim(xmax=width, xmin=0)
        plt.ylim(ymax=0, ymin=height)
        for i in range(1, width, 1):
            ax.plot([i, i], [0, height], 'm', linestyle='dotted', lw=0.5)
        for j in range(1, height, 1):
            ax.plot([0, width], [j, j], 'm', linestyle='dotted', lw=0.5)
        ax.set_title('valid region')
        valid_region_all = np.zeros([height, width])
    for gt_box in gt_boxes:
        valid_region = np.zeros([height, width])
        x1, y1 = (gt_box[:-1]/16.0)[0:2]
        x2, y2 = (gt_box[:-1]/16.0)[2:4]
        if x2 > width:
            x2 = width
        if y2 > height:
            y2 = height
        x1_floor, y1_floor = np.floor([x1, y1])
        x2_ceil, y2_ceil = np.ceil([x2, y2])
        # fill valid region
        for i in np.arange(x1_floor, x2_ceil, dtype=np.int):
            for j in np.arange(y1_floor, y2_ceil, dtype=np.int):
                # left top
                if j <= y1 and i <= x1:
                    valid_region[j, i] = (j+1-y1)*(i+1-x1)
                # top row
                elif j <= y1 and i > x1 and i+1 < x2:
                    valid_region[j, i] = (j+1-y1)*1
                # right top
                elif j <= y1 and i > x1 and i+1 >= x2:
                    valid_region[j, i] = (j+1-y1)*(x2-i)
                # top col
                elif j > y1 and i <= x1 and j+1 < y2:
                    valid_region[j, i] = 1*(i+1-x1)
                # left bottom
                elif j > y1 and i <= x1 and j+1 >= y2:
                    valid_region[j, i] = (y2-j)*(i+1-x1)
                # right bottom
                elif j+1 >= y2 and i+1 >= x2:
                    valid_region[j, i] = (y2-j)*(x2-i)
                # bottom row
                elif j+1 >= y2 and i+1 < x2 and i > x1:
                    valid_region[j, i] = (y2-j)*1
                # bottom col
                elif j+1 < y2 and i+1 >= x2 and j > y1:
                    valid_region[j, i] = 1*(x2-i)
                else:
                    valid_region[j, i] = 1.0

        # asseryt len(np.where(valid_region.ravel() != 0)[0]) == (x2_floor+1-x1_floor)*(y2_floor+1-y1_floor)
        valid_region_inds = np.where(valid_region != 0)
        valid_regions = np.hstack([valid_regions, valid_region_inds]).astype(np.int)
        if visual:
            valid_region_all += valid_region
    if visual:
        heatmap = ax.pcolor(valid_region_all, cmap=plt.cm.jet)
        plt.colorbar(heatmap)
    # plt.close('all')
    valid_regions = [np.array(valid_regions[0]), np.array(valid_regions[1])]
    return valid_regions

