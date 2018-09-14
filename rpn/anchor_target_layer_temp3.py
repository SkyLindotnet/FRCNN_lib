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
from fast_rcnn.bbox_transform import bbox_transform, keyPoints_transform, keyPoints_transform_inv
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

        anchor_scales_o = (8, 16, 32)
        anchor_scales_v1 = (4, 8, 16, 32)
        anchor_scales_v2 = (2, 4, 8, 16, 32)
        anchor_scales_v3 = (1, 2, 4, 8, 16, 32)
        anchor_scales_v4 = (0.5, 1, 2, 4, 8, 16, 32)
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
            scaleList = np.array([0.5, 1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['b', 'g', 'c', 'm', 'y', 'k', 'b']
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
            plt.title(str(anchor_scales)+'\nnum of target anchor: %s' % target_anchors.shape[0], fontsize=12)
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
            scaleList = np.array([0.5, 1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['b', 'g', 'c', 'm', 'y', 'k', 'b']
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
            plt.title(str(anchor_scales)+'\nnum of target anchor(a1_top_%d): %s' % (top_num, target_anchors.shape[0]), fontsize=12)
        plt.tight_layout()

        # add auxiliary anchors m2

        plt.figure(3)
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

            # keep only valid anchors m2
            valid_inds_t = []
            valid_inds = []
            valid_inds_s = num_anchors*width*valid_regions[0] + num_anchors*valid_regions[1]
            for valid_ind_s in valid_inds_s:
                valid_inds_t.extend(range(valid_ind_s, valid_ind_s+num_anchors))
            for valid_ind_t in valid_inds_t:
                valid_inds.extend(np.where(inds_inside == valid_ind_t)[0])
            valid_inds = np.array(valid_inds)

            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))

            # keep top n the closest anchors
            target_inds = []
            top_num = cfg.TRAIN.SELECTING_TOP_NUM
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
            ax = plt.subplot(2, 2, i+1)
            h = im_info[0]
            w = im_info[1]
            plt.xlim(xmax=w, xmin=0)
            plt.ylim(ymax=0, ymin=h)

            for i in range(16, w, 16):
                ax.plot([i, i], [0, 1000], 'm', linestyle='dotted', lw=0.5)
            for j in range(16, h, 16):
                ax.plot([0, 1000], [j, j], 'm', linestyle='dotted', lw=0.5)
            scaleList = np.array([0.5, 1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['b', 'g', 'c', 'm', 'y', 'k', 'b']
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
            plt.title(str(anchor_scales)+'\nnum of target anchor(a2_top_%d): %s' % (top_num, target_anchors.shape[0]), fontsize=12)
        plt.tight_layout()

        # add auxiliary anchors m3

        plt.figure(4)
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
                (all_anchors[:, 3] < im_info[0] + self._allowed_border)  # height
            )[0]
            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]

            # keep only valid anchors m2
            valid_inds_t = []
            valid_inds = []
            valid_inds_s = num_anchors * width * valid_regions[0] + num_anchors * valid_regions[1]
            for valid_ind_s in valid_inds_s:
                valid_inds_t.extend(range(valid_ind_s, valid_ind_s + num_anchors))
            for valid_ind_t in valid_inds_t:
                valid_inds.extend(np.where(inds_inside == valid_ind_t)[0])
            valid_inds = np.array(valid_inds)

            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))

            # keep top n the closest anchors
            target_inds = []
            top_num = cfg.TRAIN.SELECTING_TOP_NUM
            for k in range(gt_boxes.shape[0]):
                valid_overlaps = overlaps[valid_inds, k]  # ][:
                param_valid_overlaps_inds = np.where(valid_overlaps > 0.5)[0]
                if param_valid_overlaps_inds.shape[0] > top_num:
                    target_inds.extend(valid_inds[param_valid_overlaps_inds])
                else:
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
            ax = plt.subplot(2, 2, i + 1)
            h = im_info[0]
            w = im_info[1]
            plt.xlim(xmax=w, xmin=0)
            plt.ylim(ymax=0, ymin=h)

            for i in range(16, w, 16):
                ax.plot([i, i], [0, 1000], 'm', linestyle='dotted', lw=0.5)
            for j in range(16, h, 16):
                ax.plot([0, 1000], [j, j], 'm', linestyle='dotted', lw=0.5)
            scaleList = np.array([0.5, 1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['b', 'g', 'c', 'm', 'y', 'k', 'b']
            # target_anchors = anchors[np.where(max_overlaps > cfg.TRAIN.RPN_POSITIVE_OVERLAP)]
            # target_anchors = np.vstack((anchors[gt_argmax_overlaps], target_anchors))
            target_anchors = anchors[target_inds]  # valid_inds
            for anchor in target_anchors:
                anchor_w = anchor[2] - anchor[0]
                anchor_h = anchor[3] - anchor[1]
                scale = int(round(math.sqrt(anchor_w * anchor_h / 256)))
                index = np.where(scaleList == scale)[0]
                if len(index) == 0:
                    index = np.where(scaleList == (scale + 1))[0]
                if len(index) == 0:
                    index = np.where(scaleList == (scale - 1))[0]
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
            plt.title(str(anchor_scales) + '\nnum of target anchor(a3_top_%d): %s' % (top_num, target_anchors.shape[0]),
                      fontsize=12)
        plt.tight_layout()

        # print 'done'
        plt.close('all')

    def debugPyramidAnchorPosition(self, shifts, im_info, gt_boxes, valid_regions, height, width, anchor_scales, feat_stride):
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

        anchor_scales_list = [anchor_scales]

        K = shifts.shape[0]  # number of proposals is width*height
        plt.figure(1)
        for i, anchor_scales in enumerate(anchor_scales_list):
            _anchors = generate_anchors(base_size=feat_stride,
                                        scales=np.array(anchor_scales))
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

            # visual target anchor and gt
            # plt.figure(i)
            ax = plt.subplot(2, 2, 1)
            h = im_info[0]
            w = im_info[1]
            plt.xlim(xmax=w, xmin=0)
            plt.ylim(ymax=0, ymin=h)

            for i in range(feat_stride, w, feat_stride):
                ax.plot([i, i], [0, 1000], 'm', linestyle='dotted', lw=0.5)
            for j in range(feat_stride, h, feat_stride):
                ax.plot([0, 1000], [j, j], 'm', linestyle='dotted', lw=0.5)
            scaleList = np.array([0.5, 1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['b', 'g', 'c', 'm', 'y', 'k', 'b']
            target_anchors = anchors[np.where(max_overlaps > cfg.TRAIN.RPN_POSITIVE_OVERLAP)]
            target_anchors = np.vstack((anchors[gt_argmax_overlaps], target_anchors))
            for anchor in target_anchors:
                anchor_w = anchor[2] - anchor[0]
                anchor_h = anchor[3] - anchor[1]
                scale = int(round(math.sqrt(anchor_w*anchor_h/(feat_stride*feat_stride))))
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
            plt.title(str(anchor_scales)+'\nnum of target anchor: %s' % target_anchors.shape[0], fontsize=12)
        plt.tight_layout()

        # add auxiliary anchors m2
        # plt.figure(2)
        for i, anchor_scales in enumerate(anchor_scales_list):
            _anchors = generate_anchors(base_size=feat_stride,
                                        scales=np.array(anchor_scales))
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

            # keep only valid anchors m2
            valid_inds_t = []
            valid_inds = []
            valid_inds_s = num_anchors*width*valid_regions[0] + num_anchors*valid_regions[1]
            for valid_ind_s in valid_inds_s:
                valid_inds_t.extend(range(valid_ind_s, valid_ind_s+num_anchors))
            for valid_ind_t in valid_inds_t:
                valid_inds.extend(np.where(inds_inside == valid_ind_t)[0])
            valid_inds = np.array(valid_inds)

            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))

            # keep top n the closest anchors
            target_inds = []
            top_num = cfg.TRAIN.SELECTING_TOP_NUM
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
            ax = plt.subplot(2, 2, 2)
            h = im_info[0]
            w = im_info[1]
            plt.xlim(xmax=w, xmin=0)
            plt.ylim(ymax=0, ymin=h)

            for i in range(feat_stride, w, feat_stride):
                ax.plot([i, i], [0, 1000], 'm', linestyle='dotted', lw=0.5)
            for j in range(feat_stride, h, feat_stride):
                ax.plot([0, 1000], [j, j], 'm', linestyle='dotted', lw=0.5)
            scaleList = np.array([0.5, 1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['b', 'g', 'c', 'm', 'y', 'k', 'b']
            # target_anchors = anchors[np.where(max_overlaps > cfg.TRAIN.RPN_POSITIVE_OVERLAP)]
            # target_anchors = np.vstack((anchors[gt_argmax_overlaps], target_anchors))
            target_anchors = anchors[target_inds]  # valid_inds
            for anchor in target_anchors:
                anchor_w = anchor[2] - anchor[0]
                anchor_h = anchor[3] - anchor[1]
                scale = int(round(math.sqrt(anchor_w*anchor_h/(feat_stride*feat_stride))))
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
            plt.title(str(anchor_scales)+'\nnum of target anchor(a2_top_%d): %s' % (top_num, target_anchors.shape[0]), fontsize=12)
        plt.tight_layout()

        # add auxiliary anchors m3

        # plt.figure(3)
        for i, anchor_scales in enumerate(anchor_scales_list):
            _anchors = generate_anchors(base_size=feat_stride,
                                        scales=np.array(anchor_scales))
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
                (all_anchors[:, 3] < im_info[0] + self._allowed_border)  # height
            )[0]
            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]

            # keep only valid anchors m2
            valid_inds_t = []
            valid_inds = []
            valid_inds_s = num_anchors * width * valid_regions[0] + num_anchors * valid_regions[1]
            for valid_ind_s in valid_inds_s:
                valid_inds_t.extend(range(valid_ind_s, valid_ind_s + num_anchors))
            for valid_ind_t in valid_inds_t:
                valid_inds.extend(np.where(inds_inside == valid_ind_t)[0])
            valid_inds = np.array(valid_inds)

            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))

            # keep top n the closest anchors
            target_inds = []
            top_num = cfg.TRAIN.SELECTING_TOP_NUM
            for k in range(gt_boxes.shape[0]):
                valid_overlaps = overlaps[valid_inds, k]  # ][:
                param_valid_overlaps_inds = np.where(valid_overlaps > 0.5)[0]
                if param_valid_overlaps_inds.shape[0] > top_num:
                    target_inds.extend(valid_inds[param_valid_overlaps_inds])
                else:
                    sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                    target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])

            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]
            # select repetitive gt argmax overlaps
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            # visual target anchor and gt
            # plt.figure(i)
            ax = plt.subplot(2, 2, 3)
            h = im_info[0]
            w = im_info[1]
            plt.xlim(xmax=w, xmin=0)
            plt.ylim(ymax=0, ymin=h)

            for i in range(feat_stride, w, feat_stride):
                ax.plot([i, i], [0, 1000], 'm', linestyle='dotted', lw=0.5)
            for j in range(feat_stride, h, feat_stride):
                ax.plot([0, 1000], [j, j], 'm', linestyle='dotted', lw=0.5)
            scaleList = np.array([0.5, 1, 2, 4, 8, 16, 32])  # np.array(cfg.DEFALU_ANCHOR_SCALES)
            colors = ['b', 'g', 'c', 'm', 'y', 'k', 'b']
            RPN_POSITIVE_OVERLAP = 0.5  # cfg.TRAIN.RPN_POSITIVE_OVERLAP
            # target_anchors = anchors[np.where(max_overlaps > RPN_POSITIVE_OVERLAP)]
            # target_anchors = np.vstack((anchors[gt_argmax_overlaps], target_anchors))
            target_anchors = anchors[target_inds]  # valid_inds
            for anchor in target_anchors:
                anchor_w = anchor[2] - anchor[0]
                anchor_h = anchor[3] - anchor[1]
                scale = int(round(math.sqrt(anchor_w * anchor_h / (feat_stride*feat_stride))))
                index = np.where(scaleList == scale)[0]
                if len(index) == 0:
                    index = np.where(scaleList == (scale + 1))[0]
                if len(index) == 0:
                    index = np.where(scaleList == (scale - 1))[0]
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
            plt.title(str(anchor_scales) + '\nnum of target anchor(a3_top_%d): %s' % (top_num, target_anchors.shape[0]),
                      fontsize=12)
        plt.tight_layout()

        # print 'done'
        plt.close('all')

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        # anchor_scales = layer_params.get('scales', (8, 16, 32))
        if cfg.PYRAMID_MORE:
            DEFALU_ANCHOR_SCALES = cfg.PYRAMID_MORE_ANCHORS[-1]
        else:
            DEFALU_ANCHOR_SCALES = cfg.DEFALU_ANCHOR_SCALES
        anchor_scales = layer_params.get('scales', DEFALU_ANCHOR_SCALES)  # 8, 16, 32

        self._anchors = generate_anchors(scales=np.array(anchor_scales))  # generate 9 anchors with different raido and scale
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']
        self._anchor_scales = anchor_scales

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
        self._allowed_border = layer_params.get('allowed_border', cfg.TRAIN.RPN_ALLOWED_BORDER)

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

        if cfg.TRAIN.RPN_KP_REGRESSION:
            # keyPoint_targets
            keyPoint_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
            top[4].reshape(1, A * keyPoint_num, height, width)
            # keyPoint_inside_weights
            top[5].reshape(1, A * keyPoint_num, height, width)
            # keyPoint_outside_weights
            top[6].reshape(1, A * keyPoint_num, height, width)

        if cfg.PYRAMID_MORE:
            assert len(cfg.PYRAMID_MORE_ANCHORS) == len(layer_params)
            self._anchors_from_extends = []
            self._num_anchors_from_extends = []
            self._feat_stride_from_extends = []
            self._anchor_scales_from_extends = []
            PYRAMID_NUM = len(layer_params)
            for i, j in zip(range(4, 5-PYRAMID_NUM, -1), range(1, PYRAMID_NUM)):
                feat_stride_str = "feat_stride_from_p%d_3" % i
                anchor_scales_extend = layer_params.get('scales', cfg.PYRAMID_MORE_ANCHORS[PYRAMID_NUM-1-j])
                # generate anchors with different raido and scale
                self._anchors_from_extend = generate_anchors(base_size=layer_params[feat_stride_str],
                                                             scales=np.array(anchor_scales_extend))
                self._num_anchors_from_extend = self._anchors_from_extend.shape[0]
                self._feat_stride_from_extend = layer_params[feat_stride_str]
                # record info of anchors from extend convs
                self._anchors_from_extends.append(self._anchors_from_extend)
                self._num_anchors_from_extends.append(self._num_anchors_from_extend)
                self._feat_stride_from_extends.append(self._feat_stride_from_extend)
                self._anchor_scales_from_extends.append(anchor_scales_extend)

                # allow boxes to sit over the edge by a small amount
                self._allowed_border_from_extend = layer_params.get('allowed_border', cfg.TRAIN.RPN_ALLOWED_BORDER)

                height_from_extend, width_from_extend = bottom[3+j].data.shape[-2:]

                A_from_extend = self._num_anchors_from_extend
                # labels_from_extend
                top[0+j*4].reshape(1, 1, A_from_extend * height_from_extend, width_from_extend)
                # bbox_targets_from_extend
                top[1+j*4].reshape(1, A_from_extend * 4, height_from_extend, width_from_extend)
                # bbox_inside_weights_from_extend
                top[2+j*4].reshape(1, A_from_extend * 4, height_from_extend, width_from_extend)
                # bbox_outside_weights_from_extend
                top[3+j*4].reshape(1, A_from_extend * 4, height_from_extend, width_from_extend)

        if cfg.RPN_PYRAMID_MORE:
            RPN_PYRAMID_NUM = cfg.RPN_PYRAMID_NUM
            for j in range(1, RPN_PYRAMID_NUM):
                height_from_extend, width_from_extend = bottom[3+j].data.shape[-2:]

                # labels_from_extend
                top[0+j*4].reshape(1, 1, A * height_from_extend, width_from_extend)
                # bbox_targets_from_extend
                top[1+j*4].reshape(1, A * 4, height_from_extend, width_from_extend)
                # bbox_inside_weights_from_extend
                top[2+j*4].reshape(1, A * 4, height_from_extend, width_from_extend)
                # bbox_outside_weights_from_extend
                top[3+j*4].reshape(1, A * 4, height_from_extend, width_from_extend)

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
            # return valid regions according to gt_boxes
            if not cfg.PYRAMID_MORE:
                valid_regions = _compute_valid_regions(height, width, gt_boxes, 0)
            else:
                valid_regions = _compute_valid_regions_extend(height, width, gt_boxes, self._feat_stride, 0)

            # keep only valid anchors
            # timer = Timer()
            # timer.tic()
            # valid_inds = []
            # all_anchors = all_anchors.reshape((height, width, A, 4))
            # valid_anchors = all_anchors[valid_regions]
            # valid_anchors = valid_anchors.reshape((valid_anchors.shape[0]*valid_anchors.shape[1], 4))
            # for valid_anchor in valid_anchors:
            #     valid_inds = np.hstack([valid_inds, np.where((anchors == valid_anchor).all(1))[0]]).astype(np.int)
            # timer.toc()
            # print ('valid_anchors hstack took {:.3f}s').format(timer.total_time)

            # keep only valid anchors m2
            valid_inds_t = []
            valid_inds = []
            valid_inds_s = A*width*valid_regions[0] + A*valid_regions[1]
            for valid_ind_s in valid_inds_s:
                valid_inds_t.extend(range(valid_ind_s, valid_ind_s+A))
            for valid_ind_t in valid_inds_t:
                valid_inds.extend(np.where(inds_inside == valid_ind_t)[0])
            valid_inds = np.array(valid_inds)

            # keep top n the closest anchors
            target_inds = []
            top_num = cfg.TRAIN.SELECTING_TOP_NUM

            for k in range(gt_boxes.shape[0]):
                if cfg.TRAIN.FUSE_ANCHORS_STRATEGY:
                    valid_overlaps = overlaps[valid_inds, k]  # ][:
                    param_valid_overlaps_inds = np.where(valid_overlaps > cfg.TRAIN.FUSE_ANCHORS_THRESH)[0]
                    if param_valid_overlaps_inds.shape[0] > top_num:
                        target_inds.extend(valid_inds[param_valid_overlaps_inds])
                    else:
                        sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                        target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])
                else:
                    valid_overlaps = overlaps[valid_inds, k]
                    sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                    target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])

        # compute max overlap between every inside anchors and all gt
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # visual target anchor and gt
        if cfg.TRAIN.VISUAL_ANCHORS:
            if not cfg.PYRAMID_MORE:
                self.debugAnchorPosition(shifts, im_info, gt_boxes, valid_regions, height, width)
            else:
                self.debugPyramidAnchorPosition(shifts, im_info, gt_boxes, valid_regions,
                                                height, width, self._anchor_scales, self._feat_stride)
        # add scale info for visual attribute in proposal target layer
        if cfg.TRAIN.VISUAL_ATTRIBUTES:
            cfg.TRAIN.VISUAL_SCALE = im_info[2]

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

        if cfg.TRAIN.RPN_POSITION_ANCHORS_OHEM:
            # from softmax layer
            all_fg_scores = bottom[0].data[:, self._num_anchors:, :, :]
            all_fg_position = np.zeros(all_fg_scores.shape)
            for i in range(all_fg_position.shape[1]):
                all_fg_position[0, i, :, :] = i + 1
            all_fg_scores = all_fg_scores.reshape((K * A, -1))
            all_fg_position = all_fg_position.reshape((K * A, -1))
            fg_scores = all_fg_scores[inds_inside].ravel()
            fg_position = all_fg_position[inds_inside].ravel()
            # mining fg of which score is larger than threshold
            if cfg.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_FG:
                fg_inds = np.where(labels == 1)[0]
                valid_fg_scores = fg_scores[fg_inds]
                ignore_valid_fg_scores_inds = np.where(valid_fg_scores > cfg.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_FG_threshold)[0]
                ignore_fg_inds = fg_inds[ignore_valid_fg_scores_inds]
                labels[ignore_fg_inds] = -1

            if cfg.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_BG:
                bg_inds = np.where(labels == 0)[0]
                valid_bg_scores = fg_scores[bg_inds]
                valid_bg_position = fg_position[bg_inds]
                if cfg.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_BALANCE == 1:
                    fg_num = len(np.where(labels == 1)[0])
                    if fg_num < cfg.TRAIN.RPN_BATCHSIZE / 2:
                        bg_num = cfg.TRAIN.RPN_BATCHSIZE - fg_num
                    else:
                        bg_num = fg_num
                    sort_valid_bg_scores_inds = np.argsort(-1 * valid_bg_scores)
                    sort_bg_inds = bg_inds[sort_valid_bg_scores_inds]
                    ignore_bg_inds = sort_bg_inds[bg_num:]
                    labels[ignore_bg_inds] = -1
                    # print "bg:%d fg:%d" % (len(np.where(labels == 0)[0]), len(np.where(labels == 1)[0]))
                    assert len(np.where(labels == 0)[0]) + len(np.where(labels == 1)[0]) == fg_num + bg_num
                elif cfg.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_BALANCE == 2:
                    fg_num = len(np.where(labels == 1)[0])
                    bg_num = fg_num
                    if len(bg_inds) > bg_num:
                        sort_valid_bg_scores_inds = np.argsort(-1 * valid_bg_scores)
                        sort_bg_inds = bg_inds[sort_valid_bg_scores_inds]
                        ignore_bg_inds = sort_bg_inds[bg_num:]
                        labels[ignore_bg_inds] = -1
                elif cfg.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_BALANCE == 3:
                    fg_num = len(np.where(labels == 1)[0])
                    bg_num = fg_num
                    if len(bg_inds) > bg_num:
                        sort_valid_bg_scores_inds = np.argsort(-1 * valid_bg_scores)
                        sort_valid_bg_position = valid_bg_position[sort_valid_bg_scores_inds]
                        num_of_position = (bg_num + (self._num_anchors - bg_num % self._num_anchors)) / self._num_anchors
                        valid_bg_position_inds = np.array([np.where(sort_valid_bg_position == i)[0][:num_of_position] for i in range(1, self._num_anchors+1)]).ravel()
                        valid_bg_scores_inds = sort_valid_bg_scores_inds[valid_bg_position_inds]
                        valid_bg_inds = bg_inds[valid_bg_scores_inds]
                        labels[bg_inds] = -1
                        labels[valid_bg_inds] = 0
                        assert len(valid_bg_inds) == self._num_anchors * num_of_position
                else:
                    ignore_valid_bg_scores_inds = np.where(valid_bg_scores <= cfg.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_BG_threshold)[0]
                    ignore_bg_inds = bg_inds[ignore_valid_bg_scores_inds]
                    labels[ignore_bg_inds] = -1
        else:
            bg_inds = np.where(labels == 0)[0]
            fg_inds = np.where(labels == 1)[0]
            if cfg.TRAIN.RPN_POSITION_ANCHORS_NUM_ADAPT:
                fg_inds_num = len(np.where(labels == 1)[0])
                num_bg = fg_inds_num
                bg_inds = bg_inds
                if len(bg_inds) > num_bg:
                    disable_inds = npr.choice(
                        bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                    labels[disable_inds] = -1
            else:
                # subsample positive labels if we have too many
                num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
                # fg_inds = np.where(labels == 1)[0]
                if len(fg_inds) > num_fg:
                    disable_inds = npr.choice(
                        fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                    labels[disable_inds] = -1

                # subsample negative labels if we have too many
                num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
                # bg_inds = np.where(labels == 0)[0]
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

        # m1
        if cfg.TRAIN.RPN_KP_REGRESSION:
            timer = Timer()
            timer.tic()
            gt_keyPoints = bottom[4].data
            keyPoints_targets = _compute_keyPoints_targets(
            anchors, gt_keyPoints[argmax_overlaps, :], labels, gt_boxes[argmax_overlaps, :4])
            keyPoints_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
            timer.toc()
            print ('anchor target 1 took {:.3f}s').format(timer.total_time)

            keyPoint_inside_weights = np.zeros((len(inds_inside), keyPoints_num), dtype=np.float32)
            keyPoint_outside_weights = np.zeros((len(inds_inside), keyPoints_num), dtype=np.float32)
            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                # uniform weighting of examples (given non-uniform sampling)
                num_examples = np.sum(labels >= 0)
                positive_weights = np.ones((1, keyPoints_num)) * 1.0 / num_examples
                negative_weights = np.ones((1, keyPoints_num)) * 1.0 / num_examples
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
                positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                    np.sum(labels == 1))
                negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                    np.sum(labels == 0))
            # handle missed gt_keyPoints
            fgInds = np.where(labels == 1)[0]
            for ind in fgInds:
                if len(set(keyPoints_targets[ind, :])) != 2:
                    keyPoint_inside_weights[ind, :] = np.array(keyPoints_num * [1.0])
                    keyPoint_outside_weights[ind, :] = positive_weights
            # keyPoint_outside_weights[labels == 0, :] = negative_weights

        # m2
        if cfg.TRAIN.RPN_KP_REGRESSION:
            fgInds = np.where(labels == 1)[0]
            timer = Timer()
            timer.tic()
            gt_keyPoints = bottom[4].data
            keyPoints_targets_t = _compute_keyPoints_targets(
            anchors[fgInds, :], gt_keyPoints[argmax_overlaps, :][fgInds, :], labels[fgInds], gt_boxes[argmax_overlaps, :4][fgInds, :])
            keyPoints_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
            keyPoints_targets_t = _unmap(keyPoints_targets_t, len(inds_inside), fgInds, fill=0)
            timer.toc()
            print ('anchor target 2 took {:.3f}s').format(timer.total_time)

            keyPoint_inside_weights_t = np.zeros((len(inds_inside), keyPoints_num), dtype=np.float32)
            keyPoint_outside_weights_t = np.zeros((len(inds_inside), keyPoints_num), dtype=np.float32)
            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                # uniform weighting of examples (given non-uniform sampling)
                num_examples = np.sum(labels >= 0)
                positive_weights = np.ones((1, keyPoints_num)) * 1.0 / num_examples
                negative_weights = np.ones((1, keyPoints_num)) * 1.0 / num_examples
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
                positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                    np.sum(labels == 1))
                negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                    np.sum(labels == 0))
            # handle missed gt_keyPoints
            for ind in fgInds:
                if len(set(keyPoints_targets_t[ind, :])) != 2:
                    keyPoint_inside_weights_t[ind, :] = np.array(keyPoints_num * [1.0])
                    keyPoint_outside_weights_t[ind, :] = positive_weights
            # keyPoint_outside_weights[labels == 0, :] = negative_weights

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

        if cfg.TRAIN.RPN_KP_REGRESSION:
            # map up to original set of anchors
            keyPoints_targets = _unmap(keyPoints_targets, total_anchors, inds_inside, fill=0)
            keyPoint_inside_weights = _unmap(keyPoint_inside_weights, total_anchors, inds_inside, fill=0)
            keyPoint_outside_weights = _unmap(keyPoint_outside_weights, total_anchors, inds_inside, fill=0)

            # bbox_targets
            keyPoints_targets = keyPoints_targets \
                .reshape((1, height, width, A * keyPoints_num)).transpose(0, 3, 1, 2)
            top[4].reshape(*keyPoints_targets.shape)
            top[4].data[...] = keyPoints_targets

            # bbox_inside_weights
            keyPoint_inside_weights = keyPoint_inside_weights \
                .reshape((1, height, width, A * keyPoints_num)).transpose(0, 3, 1, 2)
            assert keyPoint_inside_weights.shape[2] == height
            assert keyPoint_inside_weights.shape[3] == width
            top[5].reshape(*keyPoint_inside_weights.shape)
            top[5].data[...] = keyPoint_inside_weights

            # bbox_outside_weights
            keyPoint_outside_weights = keyPoint_outside_weights \
                .reshape((1, height, width, A * keyPoints_num)).transpose(0, 3, 1, 2)
            assert keyPoint_outside_weights.shape[2] == height
            assert keyPoint_outside_weights.shape[3] == width
            top[6].reshape(*keyPoint_outside_weights.shape)
            top[6].data[...] = keyPoint_outside_weights

        if cfg.PYRAMID_MORE:
            PYRAMID_NUM = len(cfg.PYRAMID_MORE_ANCHORS)
            for i, j in zip(range(4, 5-PYRAMID_NUM, -1), range(0, PYRAMID_NUM-1)):
                feat_stride_str = "feat_stride_from_p%d_3" % i

                height, width = bottom[4+j].data.shape[-2:]

                # 1. Generate proposals from bbox deltas and shifted anchors
                shift_x = np.arange(0, width) * self._feat_stride_from_extends[j]  # self._feat_stride_from_conv4_3
                shift_y = np.arange(0, height) * self._feat_stride_from_extends[j]
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                    shift_x.ravel(), shift_y.ravel())).transpose()
                # add A anchors (1, A, 4) to
                # cell K shifts (K, 1, 4) to get
                # shift anchors (K, A, 4)
                # reshape to (K*A, 4) shifted anchors
                A = self._num_anchors_from_extends[j]  # self._num_anchors_from_conv4_3
                K = shifts.shape[0]
                all_anchors = (self._anchors_from_extends[j].reshape((1, A, 4)) +
                               shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
                all_anchors = all_anchors.reshape((K * A, 4))
                total_anchors = int(K * A)

                # only keep anchors inside the image
                inds_inside = np.where(
                    (all_anchors[:, 0] >= -self._allowed_border_from_extend) &
                    (all_anchors[:, 1] >= -self._allowed_border_from_extend) &
                    (all_anchors[:, 2] < im_info[1] + self._allowed_border_from_extend) &  # width
                    (all_anchors[:, 3] < im_info[0] + self._allowed_border_from_extend)    # height
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

                if cfg.TRAIN.EXTEND_ANCHORS_SELECTING:
                    # return valid regions according to gt_boxes
                    valid_regions = _compute_valid_regions_extend(height, width, gt_boxes, self._feat_stride_from_extends[j], 0)

                    # keep only valid anchors m2
                    valid_inds_t = []
                    valid_inds = []
                    valid_inds_s = A*width*valid_regions[0] + A*valid_regions[1]
                    for valid_ind_s in valid_inds_s:
                        valid_inds_t.extend(range(valid_ind_s, valid_ind_s+A))
                    for valid_ind_t in valid_inds_t:
                        valid_inds.extend(np.where(inds_inside == valid_ind_t)[0])
                    valid_inds = np.array(valid_inds)

                    # keep top n the closest anchors
                    target_inds = []
                    top_num = cfg.TRAIN.SELECTING_TOP_NUM

                    for k in range(gt_boxes.shape[0]):
                        if cfg.TRAIN.FUSE_ANCHORS_STRATEGY:
                            valid_overlaps = overlaps[valid_inds, k]  # ][:
                            param_valid_overlaps_inds = np.where(valid_overlaps > cfg.TRAIN.FUSE_ANCHORS_THRESH)[0]
                            if param_valid_overlaps_inds.shape[0] > top_num:
                                target_inds.extend(valid_inds[param_valid_overlaps_inds])
                            else:
                                sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                                target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])
                        else:
                            valid_overlaps = overlaps[valid_inds, k]
                            sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                            target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])

                # compute max overlap between every inside anchors and all gt
                argmax_overlaps = overlaps.argmax(axis=1)
                max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
                gt_argmax_overlaps = overlaps.argmax(axis=0)
                gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                           np.arange(overlaps.shape[1])]
                gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

                # visual target anchor and gt
                if cfg.TRAIN.VISUAL_ANCHORS:
                    self.debugPyramidAnchorPosition(shifts, im_info, gt_boxes, valid_regions,
                                                    height, width, self._anchor_scales_from_extends[j],
                                                    self._feat_stride_from_extends[j])

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

                if cfg.TRAIN.RPN_POSITION_ANCHORS_NUM_ADAPT:
                    fg_inds_num = len(np.where(labels == 1)[0])
                    num_bg = fg_inds_num
                    bg_inds = np.where(labels == 0)[0]
                    if len(bg_inds) > num_bg:
                        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                        labels[disable_inds] = -1
                else:
                    # subsample positive labels if we have too many
                    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
                    fg_inds = np.where(labels == 1)[0]
                    if len(fg_inds) > num_fg:
                        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                        labels[disable_inds] = -1

                    # subsample negative labels if we have too many
                    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
                    bg_inds = np.where(labels == 0)[0]
                    if len(bg_inds) > num_bg:
                        disable_inds = npr.choice(
                            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                        labels[disable_inds] = -1

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

                # map up to original set of anchors
                labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
                bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
                bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
                bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

                # labels_from_conv_extend
                labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
                labels = labels.reshape((1, 1, A * height, width))
                top[0+(j+1)*4].reshape(*labels.shape)
                top[0+(j+1)*4].data[...] = labels

                # bbox_targets_from_conv_extend
                bbox_targets = bbox_targets \
                    .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
                top[1+(j+1)*4].reshape(*bbox_targets.shape)
                top[1+(j+1)*4].data[...] = bbox_targets

                # bbox_inside_weights_from_conv_extend
                bbox_inside_weights = bbox_inside_weights \
                    .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
                assert bbox_inside_weights.shape[2] == height
                assert bbox_inside_weights.shape[3] == width
                top[2+(j+1)*4].reshape(*bbox_inside_weights.shape)
                top[2+(j+1)*4].data[...] = bbox_inside_weights

                # bbox_outside_weights_from_conv_extend
                bbox_outside_weights = bbox_outside_weights \
                    .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
                assert bbox_outside_weights.shape[2] == height
                assert bbox_outside_weights.shape[3] == width
                top[3+(j+1)*4].reshape(*bbox_outside_weights.shape)
                top[3+(j+1)*4].data[...] = bbox_outside_weights

        if cfg.RPN_PYRAMID_MORE:
            RPN_PYRAMID_NUM = cfg.RPN_PYRAMID_NUM
            for j in range(1, RPN_PYRAMID_NUM):

                height_extend, width_extend = bottom[3+j].data.shape[-2:]

                # 1. Generate proposals from bbox deltas and shifted anchors
                shift_x = np.arange(0, width_extend) * self._feat_stride  # self._feat_stride_from_conv4_3
                shift_y = np.arange(0, height_extend) * self._feat_stride
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                    shift_x.ravel(), shift_y.ravel())).transpose()
                # add A anchors (1, A, 4) to
                # cell K shifts (K, 1, 4) to get
                # shift anchors (K, A, 4)
                # reshape to (K*A, 4) shifted anchors
                A = self._num_anchors  # self._num_anchors_from_conv4_3
                K = shifts.shape[0]
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
                # gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
                overlaps = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(gt_boxes, dtype=np.float))

                if cfg.TRAIN.EXTEND_ANCHORS_SELECTING:
                    # return valid regions according to gt_boxes
                    valid_regions = _compute_valid_regions(height_extend, width_extend, gt_boxes, 0)

                    # keep only valid anchors m2
                    valid_inds_t = []
                    valid_inds = []
                    valid_inds_s = A*width_extend*valid_regions[0] + A*valid_regions[1]
                    for valid_ind_s in valid_inds_s:
                        valid_inds_t.extend(range(valid_ind_s, valid_ind_s+A))
                    for valid_ind_t in valid_inds_t:
                        valid_inds.extend(np.where(inds_inside == valid_ind_t)[0])
                    valid_inds = np.array(valid_inds)

                    # keep top n the closest anchors
                    target_inds = []
                    top_num = cfg.TRAIN.SELECTING_TOP_NUM

                    for k in range(gt_boxes.shape[0]):
                        if cfg.TRAIN.FUSE_ANCHORS_STRATEGY:
                            valid_overlaps = overlaps[valid_inds, k]  # ][:
                            param_valid_overlaps_inds = np.where(valid_overlaps > cfg.TRAIN.FUSE_ANCHORS_THRESH)[0]
                            if param_valid_overlaps_inds.shape[0] > top_num:
                                target_inds.extend(valid_inds[param_valid_overlaps_inds])
                            else:
                                sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                                target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])
                        else:
                            valid_overlaps = overlaps[valid_inds, k]
                            sorted_valid_overlaps_inds = np.argsort(-valid_overlaps)
                            target_inds.extend(valid_inds[sorted_valid_overlaps_inds[:top_num]])

                # compute max overlap between every inside anchors and all gt
                argmax_overlaps = overlaps.argmax(axis=1)
                max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
                gt_argmax_overlaps = overlaps.argmax(axis=0)
                gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                           np.arange(overlaps.shape[1])]
                gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

                # visual target anchor and gt
                if cfg.TRAIN.VISUAL_ANCHORS:
                    self.debugAnchorPosition(shifts, im_info, gt_boxes, valid_regions, height_extend, width_extend)
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

                if cfg.TRAIN.RPN_POSITION_ANCHORS_NUM_ADAPT:
                    fg_inds_num = len(np.where(labels == 1)[0])
                    num_bg = fg_inds_num
                    bg_inds = np.where(labels == 0)[0]
                    if len(bg_inds) > num_bg:
                        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                        labels[disable_inds] = -1
                else:
                    # subsample positive labels if we have too many
                    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
                    fg_inds = np.where(labels == 1)[0]
                    if len(fg_inds) > num_fg:
                        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                        labels[disable_inds] = -1

                    # subsample negative labels if we have too many
                    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
                    bg_inds = np.where(labels == 0)[0]
                    if len(bg_inds) > num_bg:
                        disable_inds = npr.choice(
                            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                        labels[disable_inds] = -1

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

                # map up to original set of anchors
                labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
                bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
                bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
                bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

                # labels_from_conv_extend
                labels = labels.reshape((1, height_extend, width_extend, A)).transpose(0, 3, 1, 2)
                labels = labels.reshape((1, 1, A * height_extend, width_extend))
                top[0+j*4].reshape(*labels.shape)
                top[0+j*4].data[...] = labels

                # bbox_targets_from_conv_extend
                bbox_targets = bbox_targets \
                    .reshape((1, height_extend, width_extend, A * 4)).transpose(0, 3, 1, 2)
                top[1+j*4].reshape(*bbox_targets.shape)
                top[1+j*4].data[...] = bbox_targets

                # bbox_inside_weights_from_conv_extend
                bbox_inside_weights = bbox_inside_weights \
                    .reshape((1, height_extend, width_extend, A * 4)).transpose(0, 3, 1, 2)
                assert bbox_inside_weights.shape[2] == height_extend
                assert bbox_inside_weights.shape[3] == width_extend
                top[2+j*4].reshape(*bbox_inside_weights.shape)
                top[2+j*4].data[...] = bbox_inside_weights

                # bbox_outside_weights_from_conv_extend
                bbox_outside_weights = bbox_outside_weights \
                    .reshape((1, height_extend, width_extend, A * 4)).transpose(0, 3, 1, 2)
                assert bbox_outside_weights.shape[2] == height_extend
                assert bbox_outside_weights.shape[3] == width_extend
                top[3+j*4].reshape(*bbox_outside_weights.shape)
                top[3+j*4].data[...] = bbox_outside_weights

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

def _compute_keyPoints_targets(ex_rois, gt_keyPoints, labels=None, gt_boxes=None):
    """Compute keyPoints regression targets for an image."""
    assert ex_rois.shape[1] == 4
    assert gt_keyPoints.shape[1] == cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']

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

def visual_attribute(scale, gt_boxes, gt_keyPoints):
    if cfg.TRAIN.VISUAL_ANCHORS_IMG != '':
        plt.figure()
        ax = plt.subplot(1, 2, 1)
        im = cv2.imread(cfg.TRAIN.VISUAL_ANCHORS_IMG)
        if cfg.TRAIN.VISUAL_ANCHORS_IMG_Flipped:
            im = im[:, ::-1, :]
        ax.imshow(im[:, :, ::-1], aspect='equal')
        ax = plt.subplot(1, 2, 2)
        im = cv2.resize(im, None, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR)
        ax.imshow(im[:, :, ::-1], aspect='equal')
        for gt_boxe in gt_boxes:
            gt_boxe_w = gt_boxe[2] - gt_boxe[0]
            gt_boxe_h = gt_boxe[3] - gt_boxe[1]
            rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                            ec='r', fill=False, lw=1.5)
            ax.add_patch(rec)

        gt_keyPoints = np.array(gt_keyPoints).reshape((-1, 2))
        plt.plot(gt_keyPoints[:, 0], gt_keyPoints[:, 1], 'go', ms=1.5, alpha=1)
        plt.close('all')

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

def _compute_valid_regions_extend(height, width, gt_boxes, feat_stride, visual=0):
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
        x1, y1 = (gt_box[:-1]/float(feat_stride))[0:2]
        x2, y2 = (gt_box[:-1]/float(feat_stride))[2:4]
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
