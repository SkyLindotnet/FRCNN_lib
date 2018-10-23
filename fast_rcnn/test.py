# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, \
    keyPoints_transform_inv, kp_map_transform_inv_cls, kp_map_transform_inv_reg
import argparse
from utils.timer import Timer
from utils.net_visual import vis_square
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import shutil
from mylab.draw import *

def mk_dir(saveDir, cover=1):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    elif cover:
        shutil.rmtree(saveDir)
        os.makedirs(saveDir)

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []
    if not cfg.TEST.UNSCALE:
        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)
    else:
        if im_size_max > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        else:
            im_scale = 1.0
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)


    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_image_blob_ron(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape[0:2]
    # im_size_min = np.min(im_shape[0:2])
    # im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []
    for target_size in cfg.TEST.RON_SCALES:
        # im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        im = cv2.resize(im_orig, None, None, fx=float(target_size) / im_shape[1], fy=float(target_size) / im_shape[0],
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(-1)  # im_scale = -1
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    # if cfg.ENABLE_RON:
    #     blobs['data'], im_scale_factors = _get_image_blob_ron(im)
    # else:
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors


def im_detect(net, im, boxes=None, includeRPN=False):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()
    print ('forward took {:.3f}s').format(timer.total_time)

    if cfg.TEST.VISUAL_FEATURE:
        vis_square(net.blobs['conv1_1'].data[0])
        vis_square(net.blobs['conv2_1'].data[0, :36])
        vis_square(net.blobs['conv3_1'].data[0, :36])

    if cfg.TEST.PRINT_FUSE_WEIGHT:
        print_fuse_weight(net)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                rois_from_p4_3 = net.blobs['rois_from_p4_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
            else:
                rois_from_p4_3 = net.blobs['rois_from_p4_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
                rois_from_p3_3 = net.blobs['rois_from_p3_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p3_3 = rois_from_p3_3[:, 1:5] / im_scales[0]

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                rois_from_p4_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
            else:
                rois_from_p4_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
                rois_from_p3_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p3_3 = rois_from_p3_3[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        if blobs_out.has_key('cls_prob'):
            scores = blobs_out['cls_prob']
        else:
            scores = net.blobs['cls_prob']

        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores = np.vstack((scores, scores_from_p4_3))
            else:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores_from_p3_3 = blobs_out['cls_prob_from_p3_3']
                scores = np.vstack((scores, scores_from_p4_3, scores_from_p3_3))

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores = np.vstack((scores, scores_from_p4_3))
            else:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores_from_p3_3 = blobs_out['cls_prob_from_p3_3']
                scores = np.vstack((scores, scores_from_p4_3, scores_from_p3_3))

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        if blobs_out.has_key('bbox_pred'):
            box_deltas = blobs_out['bbox_pred']
        else:
            box_deltas = net.blobs['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3))
            else:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                box_deltas_from_p3_3 = blobs_out['bbox_pred_from_p3_3']
                pred_boxes_from_p3_3 = bbox_transform_inv(boxes_from_p3_3, box_deltas_from_p3_3)
                pred_boxes_from_p3_3 = clip_boxes(pred_boxes_from_p3_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3, pred_boxes_from_p3_3))

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3))
            else:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                box_deltas_from_p3_3 = blobs_out['bbox_pred_from_p3_3']
                pred_boxes_from_p3_3 = bbox_transform_inv(boxes_from_p3_3, box_deltas_from_p3_3)
                pred_boxes_from_p3_3 = clip_boxes(pred_boxes_from_p3_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3, pred_boxes_from_p3_3))

    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
    if includeRPN:
        scoresRPN = net.blobs['rpn_scores'].data.copy()
        boxesRPN = boxes
        return scores, pred_boxes, scoresRPN, boxesRPN
    else:
        return scores, pred_boxes


def im_detect_by_rois(net, im, boxes=None, rois_layer = 'rois'):
    """Generate RPN proposals on a single image."""
    # blobs = {}
    # blobs['data'], blobs['im_info'] = _get_image_blob(im)
    # net.blobs['data'].reshape(*(blobs['data'].shape))
    # net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    # timer = Timer()
    # timer.tic()
    # blobs_out = net.forward(
    #         data=blobs['data'].astype(np.float32, copy=False),
    #         im_info=blobs['im_info'].astype(np.float32, copy=False))
    # timer.toc()
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs[rois_layer] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs[rois_layer] = blobs[rois_layer][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs[rois_layer].reshape(*(blobs[rois_layer].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs[rois_layer] = blobs[rois_layer].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()

    print ('forward took {:.3f}s').format(timer.total_time)

    scale = blobs['im_info'][0, 2]
    # boxes = blobs_out['rois'][:, 1:].copy() / scale  # blobs_out
    # scores = blobs_out['scores'].copy()
    boxes = net.blobs[rois_layer].data[:, 1:].copy() / scale  # blobs_out
    if rois_layer == 'rois':
        scores = net.blobs['rpn_scores'].data.copy()
    elif rois_layer == 'fc_rois':
        scores = net.blobs['fc_scores'].data.copy()

    return scores, boxes


def im_detect_facePlus(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()
    print ('forward took {:.3f}s').format(timer.total_time)
    if cfg.TEST.VISUAL_FEATURE:
        vis_square(net.blobs['conv1_1'].data[0])
        vis_square(net.blobs['conv2_1'].data[0, :36])
        vis_square(net.blobs['conv3_1'].data[0, :36])

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                rois_from_p4_3 = net.blobs['rois_from_p4_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
            else:
                rois_from_p4_3 = net.blobs['rois_from_p4_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
                rois_from_p3_3 = net.blobs['rois_from_p3_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p3_3 = rois_from_p3_3[:, 1:5] / im_scales[0]

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                rois_from_p4_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
            else:
                rois_from_p4_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
                rois_from_p3_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p3_3 = rois_from_p3_3[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores = np.vstack((scores, scores_from_p4_3))
            else:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores_from_p3_3 = blobs_out['cls_prob_from_p3_3']
                scores = np.vstack((scores, scores_from_p4_3, scores_from_p3_3))

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores = np.vstack((scores, scores_from_p4_3))
            else:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores_from_p3_3 = blobs_out['cls_prob_from_p3_3']
                scores = np.vstack((scores, scores_from_p4_3, scores_from_p3_3))

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3))
            else:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                box_deltas_from_p3_3 = blobs_out['bbox_pred_from_p3_3']
                pred_boxes_from_p3_3 = bbox_transform_inv(boxes_from_p3_3, box_deltas_from_p3_3)
                pred_boxes_from_p3_3 = clip_boxes(pred_boxes_from_p3_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3, pred_boxes_from_p3_3))

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3))
            else:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                box_deltas_from_p3_3 = blobs_out['bbox_pred_from_p3_3']
                pred_boxes_from_p3_3 = bbox_transform_inv(boxes_from_p3_3, box_deltas_from_p3_3)
                pred_boxes_from_p3_3 = clip_boxes(pred_boxes_from_p3_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3, pred_boxes_from_p3_3))

    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    # add keyPoints out keyPoint_pred
    if cfg.TRAIN.RPN_KP_REGRESSION:
        rois_keyPoints = blobs_out['rois_keyPoints']
        # unscale back to raw image space
        rois_keyPoints = rois_keyPoints[:, 1:] / im_scales[0]
        keyPoint_deltas = blobs_out['keyPoint_pred']
        keyPoint_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
        keyPoint_deltas = keyPoint_deltas[:, keyPoint_num:]
        pred_keyPoints = keyPoints_transform_inv(boxes, keyPoint_deltas, rois_keyPoints, 'elewise')
    else:
        keyPoint_deltas = blobs_out['keyPoint_pred']
        keyPoint_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
        keyPoint_deltas = keyPoint_deltas[:, keyPoint_num:]
        pred_keyPoints = keyPoints_transform_inv(boxes, keyPoint_deltas)

    return scores, pred_boxes, pred_keyPoints


def im_detect_kp_by_rois(net, im, boxes=None, rois_layer = 'rois'):
    """Generate RPN proposals on a single image."""
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs[rois_layer] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs[rois_layer] = blobs[rois_layer][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs[rois_layer].reshape(*(blobs[rois_layer].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs[rois_layer] = blobs[rois_layer].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()

    print ('forward took {:.3f}s').format(timer.total_time)

    if cfg.TEST.PRINT_FUSE_WEIGHT:
        print_fuse_weight(net)

    scale = blobs['im_info'][0, 2]
    # boxes = blobs_out['rois'][:, 1:].copy() / scale  # blobs_out
    # scores = blobs_out['scores'].copy()
    boxes = net.blobs[rois_layer].data[:, 1:].copy() / scale  # blobs_out
    if rois_layer == 'rois':
        scores = net.blobs['rpn_scores'].data.copy()
    elif rois_layer == 'fc_rois':
        scores = net.blobs['fc_scores'].data.copy()

    keyPoint_deltas = blobs_out['kp_pred']  # keyPoint_pred
    keyPoint_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
    keyPoint_deltas = keyPoint_deltas[:, keyPoint_num:]
    pred_keyPoints = keyPoints_transform_inv(boxes, keyPoint_deltas)

    return scores, boxes, pred_keyPoints


def im_detect_kp_map_by_rois(net, im, boxes=None, rois_layer='rois'):
    """Generate RPN proposals on a single image."""
    blobs, im_scales = _get_blobs(im, boxes)
    cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE = im_scales[0]
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs[rois_layer] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs[rois_layer] = blobs[rois_layer][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs[rois_layer].reshape(*(blobs[rois_layer].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs[rois_layer] = blobs[rois_layer].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()

    print ('forward took {:.3f}s').format(timer.total_time)

    scale = blobs['im_info'][0, 2]
    # boxes = blobs_out['rois'][:, 1:].copy() / scale  # blobs_out
    # scores = blobs_out['scores'].copy()
    # if cfg.ENABLE_RON:
    #     im_shape = im.shape[0:2]
    #     im_scales = float(cfg.TEST.RON_SCALES[0]) / np.array(im_shape)
    #     boxes = net.blobs[rois_layer].data[:, 1:].copy()
    #     boxes[:, 0::2] = boxes[:, 0::2] / im_scales[1]
    #     boxes[:, 1::2] = boxes[:, 1::2] / im_scales[0]
    # else:
    boxes = net.blobs[rois_layer].data[:, 1:].copy() / scale  # blobs_out
    if rois_layer == 'rois':
        scores = net.blobs['rpn_scores'].data.copy()
    elif rois_layer == 'fc_rois':
        scores = net.blobs['fc_scores'].data.copy()

    if cfg.TRAIN.PREDICE_KP_REGRESSION:
        keyPoint_map = blobs_out['kp_score']
    else:
        keyPoint_map = blobs_out['kp_prob']

    kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] / 2
    pred_keyPoints = np.zeros([keyPoint_map.shape[0], kp_num*2])
    # score of keymap
    pred_key_scores = np.zeros([keyPoint_map.shape[0], kp_num])

    # visual maps
    # dic= {}
    # dic['kp_deconv1'] = np.mean(net.blobs['kp_deconv1'].data[0], 0)
    # dic['kp_deconv2'] = np.mean(net.blobs['kp_deconv2'].data[0], 0)
    # dic['kp_deconv3'] = np.mean(net.blobs['kp_deconv3'].data[0], 0)
    # dic['kp_score_bg'] = net.blobs['kp_score'].data[0][0]
    # dic['kp_score_5'] = net.blobs['kp_score'].data[0][5]
    # dic['kp_score_15'] = net.blobs['kp_score'].data[0][15]
    # dic['kp_prob_bg'] = net.blobs['kp_prob'].data[0][0]
    # dic['kp_prob_5'] = net.blobs['kp_prob'].data[0][5]
    # dic['kp_prob_15'] = net.blobs['kp_prob'].data[0][15]

    # dic['bf_score_bg'] = net.blobs['bf_score'].data[0][0]
    # dic['bf_score_fg'] = net.blobs['bf_score'].data[0][1]
    # dic['bf_score_mask_fg'] = net.blobs['bf_score_mask'].data[0][1]
    # dic['kp_score_fusion_bg'] = net.blobs['kp_score_fusion'].data[0][0]
    # dic['kp_score_fusion_5'] = net.blobs['kp_score_fusion'].data[0][5]
    # dic['kp_score_fusion_15'] = net.blobs['kp_score_fusion'].data[0][15]
    # dic['kp_score_attention_bg'] = net.blobs['kp_score_attention'].data[0][0]
    # dic['kp_score_attention_5'] = net.blobs['kp_score_attention'].data[0][5]
    # dic['kp_score_attention_15'] = net.blobs['kp_score_attention'].data[0][15]

    # dic['bf_score_bg'] = net.blobs['bf_score'].data[0][0]
    # dic['bf_score_fg'] = net.blobs['bf_score'].data[0][1]
    # dic['bf_score_mask_bg'] = net.blobs['bf_score_mask'].data[0][0]
    # dic['bf_score_mask_fg'] = net.blobs['bf_score_mask'].data[0][1]
    # dic['kp_score_bg'] = net.blobs['kp_score'].data[0][0]
    # dic['kp_score_5'] = net.blobs['kp_score'].data[0][5]
    # dic['kp_score_mask_bg'] = net.blobs['kp_score_mask'].data[0][0]
    # dic['kp_score_mask_5'] = net.blobs['kp_score_mask'].data[0][5]
    # dic['kp_score_attention_bg'] = net.blobs['kp_score_attention'].data[0][0]
    # dic['kp_score_attention_5'] = net.blobs['kp_score_attention'].data[0][5]
    # dic['kp_prob_bg'] = net.blobs['kp_prob'].data[0][0]
    # dic['kp_prob_5'] = net.blobs['kp_prob'].data[0][5]
    # dic['kp_deconv1'] = np.mean(net.blobs['kp_deconv1'].data[0], 0)
    # dic['kp_deconv2'] = np.mean(net.blobs['kp_deconv2'].data[0], 0)
    # dic['kp_deconv3'] = np.mean(net.blobs['kp_deconv3'].data[0], 0)
    # dic['kp_score_bg'] = net.blobs['kp_score'].data[0][0]
    # dic['kp_score_fg'] = np.sum(net.blobs['kp_score'].data[0][1:], 0)
    # dic['kp_prob_bg'] = net.blobs['kp_prob'].data[0][0]
    # dic['kp_prob_fg'] = np.sum(net.blobs['kp_prob'].data[0][1:], 0)

    for i, boxe in enumerate(boxes):
        # if scores[i] > 0.95:
        strategy = 2  # 1: select max from 68 kp_maps 2: selece max from every kp_map
        if not cfg.TRAIN.MAP_ONLY_FG:
            if strategy == 1:
                arg_out = keyPoint_map[i][1:].argmax(axis=0)
                out = keyPoint_map[i][1:].max(axis=0)
                kp_map = np.zeros([out.shape[0], out.shape[0]])
                for j in range(kp_num):
                    arg_out_kp = np.where(arg_out == j)
                    if len(arg_out_kp[0]) != 0:
                        out_kp = out[arg_out_kp]
                        # visual map
                        # v_map = np.zeros(out.shape[0], out.shape[1])
                        # v_map[arg_out_kp] = out_kp
                        # visual_kp_map(v_map)
                        arg_kp = out_kp.argmax(axis=0)
                        kp_map[arg_out_kp[0][arg_kp], arg_out_kp[1][arg_kp]] = j + 1
                if cfg.TRAIN.PREDICE_KP_REGRESSION:
                    gt_keyPoints_inv = kp_map_transform_inv_reg(boxe, kp_map, cfg.TRAIN.MAP_offset)
                else:
                    gt_keyPoints_inv = kp_map_transform_inv_cls(boxe, kp_map, cfg.TRAIN.MAP_offset)
                pred_keyPoints[i] = gt_keyPoints_inv
            elif strategy == 2:
                if cfg.TRAIN.PREDICE_KP_REGRESSION:
                    kp_maps = np.zeros([kp_num, keyPoint_map.shape[2] * keyPoint_map.shape[3], 1], dtype=np.int)
                    for j in range(0, kp_num):
                        kp_map = keyPoint_map[i][j]
                        arg_out = kp_map.argmax()
                        out = kp_map.max()
                        # visual map
                        # visual_kp_map(kp_map)
                        kp_maps[j][arg_out] = 1
                    kp_maps = kp_maps.reshape([kp_num, keyPoint_map.shape[2], keyPoint_map.shape[3]])
                    # visual_kp_maps(kp_maps)
                    gt_keyPoints_inv = kp_map_transform_inv_reg(boxe, kp_maps, cfg.TRAIN.MAP_offset)
                else:
                    kp_maps = np.zeros([keyPoint_map.shape[2] * keyPoint_map.shape[3], 1], dtype=np.int)
                    re_dict = {}
                    for j in range(1, kp_num+1):
                        kp_map = keyPoint_map[i][j]
                        arg_out = kp_map.argmax()
                        out = kp_map.max()
                        # visual map
                        # visual_kp_map(kp_map)
                        pred_key_scores[i][j-1] = out
                        kp_maps[arg_out] = j
                        temps = kp_maps.reshape([keyPoint_map.shape[2], keyPoint_map.shape[3]])
                        re_dict[j] = np.where(temps == j)  # avoid repeating
                    kp_maps = kp_maps.reshape([keyPoint_map.shape[2], keyPoint_map.shape[3]])
                    gt_keyPoints_inv = kp_map_transform_inv_cls(boxe, kp_maps, cfg.TRAIN.MAP_offset, re_dict)
                    # visual_kp_debug(im, gt_keyPoints_inv, boxe)
                    # if sum([x in cfg.TRAIN.VISUAL_ANCHORS_IMG for x in
                    #         ['indoor_110', 'indoor_214', 'outdoor_051', 'outdoor_096', 'outdoor_155', 'outdoor_281']]):
                    #     bf_map = net.blobs['bf_score_mask'].data[i][1]
                    #     kp_afer_mask = np.sum(keyPoint_map[i][1:], 0)
                    #     kp_before_mask = np.sum(blobs_out['kp_before_mask'][i][1:], 0)
                    #     visual_bf_test(im, boxe, bf_map, kp_before_mask, kp_afer_mask, gt_keyPoints_inv,
                    #                    cfg.TEST.IMAGE_SAVE_DIR)

                pred_keyPoints[i] = gt_keyPoints_inv
            elif strategy == 3:
                joint_map = np.ones([keyPoint_map.shape[2], keyPoint_map.shape[3]])
                for k in range(0, kp_num+1):
                    joint_map = joint_map * (1 - keyPoint_map[i][k])
                kp_maps = np.zeros([keyPoint_map.shape[2]*keyPoint_map.shape[3], 1], dtype=np.int)
                for j in range(1, kp_num+1):
                    kp_map = joint_map/(1-keyPoint_map[i][j])*keyPoint_map[i][j]
                    arg_out = kp_map.argmax()
                    out = kp_map.max()
                    # visual map
                    # visual_kp_map(kp_map)
                    kp_maps[arg_out] = j
                kp_maps = kp_maps.reshape([keyPoint_map.shape[2], keyPoint_map.shape[3]])
                # visual_kp_maps(kp_maps)
                if cfg.TRAIN.PREDICE_KP_REGRESSION:
                    gt_keyPoints_inv = kp_map_transform_inv_reg(boxe, kp_maps, cfg.TRAIN.MAP_offset)
                else:
                    gt_keyPoints_inv = kp_map_transform_inv_cls(boxe, kp_maps, cfg.TRAIN.MAP_offset)
                pred_keyPoints[i] = gt_keyPoints_inv
        else:
            arg_out = keyPoint_map[i][0:].argmax(axis=0)
            out = keyPoint_map[i][0:].max(axis=0)
            kp_map = np.zeros([out.shape[0], out.shape[0]])
            kp_map.fill(-1)
            for j in range(kp_num):
                arg_out_kp = np.where(arg_out == j)
                if len(arg_out_kp[0]) != 0:
                    out_kp = out[arg_out_kp]
                    arg_kp = out_kp.argmax(axis=0)
                    kp_map[arg_out_kp[0][arg_kp], arg_out_kp[1][arg_kp]] = j

            if cfg.TRAIN.PREDICE_KP_REGRESSION:
                gt_keyPoints_inv = kp_map_transform_inv_reg(boxe, kp_map, cfg.TRAIN.MAP_offset)
            else:
                gt_keyPoints_inv = kp_map_transform_inv_cls(boxe, kp_map, cfg.TRAIN.MAP_offset)
            pred_keyPoints[i] = gt_keyPoints_inv

    # visual_kp_boxes(im, boxes, scores, keyPoint_map, pred_keyPoints, cfg.TRAIN.MAP_offset)
    # visual_fpn_convs(im, net.blobs)
    # visual_convs(im, net.blobs)

    # boxes_rpn = net.blobs['rois'].data[:, 1:].copy() / scale
    # scores_rpn = net.blobs['rpn_scores'].data.copy()
    # visual_demos(im, boxes, scores, boxes_rpn, scores_rpn, pred_keyPoints)

    return scores, boxes, pred_keyPoints, pred_key_scores

def visual_kp_map(kp_map):
    ax = plt.subplot(1, 1, 1)
    heatmap = ax.imshow(kp_map, aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('segImg')
    # plt.close('all')

def visual_kp_maps(kp_maps):
    resultLabelIm = label2img(kp_maps)
    ax = plt.subplot(1, 1, 1)
    ax.imshow(resultLabelIm, aspect='equal')
    ax.set_title('segImg')
    plt.close('all')

def visual_kp_debug(im, kps, box):
    f = plt.figure(figsize=(10, 6))
    subplot = f.add_subplot(111)
    plt.imshow(im[:, :, ::-1])
    kps = kps.reshape(-1, 2)
    plt.plot(kps[:, 0], kps[:, 1], 'go', ms=1.5, alpha=1)
    draw_bbox(subplot, box)
    plt.close('all')

def visual_bf_test(im, gt_boxe, bf_map, kp_before_mask, kp_after_mask, gt_keyPoints_inv, imageSaveDir=''):
    ax = plt.subplot(2, 3, 1)
    ax.imshow(im[:, :, ::-1], aspect='equal')
    ax.set_title('Img')
    gt_boxe_w = gt_boxe[2] - gt_boxe[0]
    gt_boxe_h = gt_boxe[3] - gt_boxe[1]
    rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                    ec='r', fill=False, lw=1.5)
    ax.add_patch(rec)
    ax = plt.subplot(2, 3, 2)
    heatmap = ax.imshow(bf_map, aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('bf_map')
    ax = plt.subplot(2, 3, 3)
    heatmap = ax.imshow(kp_before_mask, aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_before_mask')
    ax = plt.subplot(2, 3, 4)
    heatmap = ax.imshow(kp_after_mask, aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_after_mask')
    ax = plt.subplot(2, 3, 5)
    ax.imshow(im[:, :, ::-1], aspect='equal')
    gt_boxe_w = gt_boxe[2] - gt_boxe[0]
    gt_boxe_h = gt_boxe[3] - gt_boxe[1]
    rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                    ec='r', fill=False, lw=1.5)
    ax.add_patch(rec)
    gt_keyPoints = np.array(gt_keyPoints_inv).reshape((-1, 2))
    ax.plot(gt_keyPoints[:, 0], gt_keyPoints[:, 1], 'go', ms=1.5, alpha=1)
    ax.set_title('gt_keyPoints')
    if imageSaveDir != '':
        mk_dir(imageSaveDir, 0)
        imfile = imageSaveDir + '/vis_' + '_'.join(cfg.TRAIN.VISUAL_ANCHORS_IMG.split('/')[-3:])
        plt.savefig(imfile, dpi=100)
    plt.close('all')

def visual_kp_maps_e1(dic):
    ax = plt.subplot(3, 3, 1)
    heatmap = ax.imshow(dic['kp_deconv1'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv1')
    ax = plt.subplot(3, 3, 2)
    heatmap = ax.imshow(dic['kp_deconv2'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv2')
    ax = plt.subplot(3, 3, 3)
    heatmap = ax.imshow(dic['kp_deconv3'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv3')

    ax = plt.subplot(3, 3, 4)
    heatmap = ax.imshow(dic['kp_score_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_bg')
    ax = plt.subplot(3, 3, 5)
    heatmap = ax.imshow(dic['kp_score_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_5')
    ax = plt.subplot(3, 3, 6)
    heatmap = ax.imshow(dic['kp_score_15'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_15')

    ax = plt.subplot(3, 3, 7)
    heatmap = ax.imshow(dic['kp_prob_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_prob_bg')
    ax = plt.subplot(3, 3, 8)
    heatmap = ax.imshow(dic['kp_prob_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_prob_5')
    ax = plt.subplot(3, 3, 9)
    heatmap = ax.imshow(dic['kp_prob_15'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_prob_15')
    # plt.close('all')

def visual_kp_maps_e2(dic):
    plt.figure(0)
    ax = plt.subplot(3, 5, 1)
    heatmap = ax.imshow(dic['kp_deconv1'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv1')
    ax = plt.subplot(3, 5, 2)
    heatmap = ax.imshow(dic['kp_deconv2'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv2')
    ax = plt.subplot(3, 5, 3)
    heatmap = ax.imshow(dic['kp_deconv3'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv3')

    ax = plt.subplot(3, 5, 4)
    heatmap = ax.imshow(dic['kp_score_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_bg')
    ax = plt.subplot(3, 5, 5)
    heatmap = ax.imshow(dic['kp_score_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_5')
    ax = plt.subplot(3, 5, 6)
    heatmap = ax.imshow(dic['kp_score_15'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_15')

    ax = plt.subplot(3, 5, 7)
    heatmap = ax.imshow(dic['bf_score_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('bf_score_bg')
    ax = plt.subplot(3, 5, 8)
    heatmap = ax.imshow(dic['bf_score_fg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('bf_score_fg')
    ax = plt.subplot(3, 5, 9)
    heatmap = ax.imshow(dic['bf_score_mask_fg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('bf_score_mask_fg')

    ax = plt.subplot(3, 5, 10)
    heatmap = ax.imshow(dic['kp_score_attention_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_attention_bg')
    ax = plt.subplot(3, 5, 11)
    heatmap = ax.imshow(dic['kp_score_attention_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_attention_5')
    ax = plt.subplot(3, 5, 12)
    heatmap = ax.imshow(dic['kp_score_attention_15'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_attention15')

    ax = plt.subplot(3, 5, 13)
    heatmap = ax.imshow(dic['kp_score_fusion_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_fusion_bg')
    ax = plt.subplot(3, 5, 14)
    heatmap = ax.imshow(dic['kp_score_fusion_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_fusion_5')
    ax = plt.subplot(3, 5, 15)
    heatmap = ax.imshow(dic['kp_score_fusion_15'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_fusion_15')
    # plt.tight_layout()
    # plt.close('all')

def visual_kp_maps_e3(dic):
    plt.figure(0)
    ax = plt.subplot(3, 5, 1)
    heatmap = ax.imshow(dic['kp_deconv1'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv1')
    ax = plt.subplot(3, 5, 2)
    heatmap = ax.imshow(dic['kp_deconv2'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv2')
    ax = plt.subplot(3, 5, 3)
    heatmap = ax.imshow(dic['kp_deconv3'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv3')

    ax = plt.subplot(3, 5, 4)
    heatmap = ax.imshow(dic['kp_score_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_bg')
    ax = plt.subplot(3, 5, 5)
    heatmap = ax.imshow(dic['kp_score_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_5')
    ax = plt.subplot(3, 5, 6)
    heatmap = ax.imshow(dic['kp_score_mask_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_mask_bg')
    ax = plt.subplot(3, 5, 7)
    heatmap = ax.imshow(dic['kp_score_mask_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_mask_5')
    ax = plt.subplot(3, 5, 8)
    heatmap = ax.imshow(dic['bf_score_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('bf_score_bg')
    ax = plt.subplot(3, 5, 9)
    heatmap = ax.imshow(dic['bf_score_fg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('bf_score_fg')
    ax = plt.subplot(3, 5, 10)
    heatmap = ax.imshow(dic['bf_score_mask_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('bf_score_mask_bg')
    ax = plt.subplot(3, 5, 11)
    heatmap = ax.imshow(dic['bf_score_mask_fg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('bf_score_mask_fg')

    ax = plt.subplot(3, 5, 12)
    heatmap = ax.imshow(dic['kp_score_attention_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_attention_bg')
    ax = plt.subplot(3, 5, 13)
    heatmap = ax.imshow(dic['kp_score_attention_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_attention_5')
    ax = plt.subplot(3, 5, 14)
    heatmap = ax.imshow(dic['kp_prob_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_prob_bg')
    ax = plt.subplot(3, 5, 15)
    heatmap = ax.imshow(dic['kp_prob_5'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_prob_5')
    # plt.tight_layout()
    plt.close('all')

def visual_kp_maps_e4(dic):
    plt.figure(0)
    ax = plt.subplot(2, 3, 1)
    heatmap = ax.imshow(dic['conv3_3'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('conv3_3')
    ax = plt.subplot(2, 3, 2)
    heatmap = ax.imshow(dic['conv4_3'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('conv4_3')
    ax = plt.subplot(2, 3, 3)
    heatmap = ax.imshow(dic['conv5_3'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('conv5_3')

    ax = plt.subplot(2, 3, 4)
    heatmap = ax.imshow(dic['kp_deconv1'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv1')
    ax = plt.subplot(2, 3, 5)
    heatmap = ax.imshow(dic['kp_deconv2'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv2')
    ax = plt.subplot(2, 3, 6)
    heatmap = ax.imshow(dic['kp_deconv3'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_deconv3')

    plt.figure(1)
    ax = plt.subplot(2, 2, 1)
    heatmap = ax.imshow(dic['kp_score_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_bg')
    ax = plt.subplot(2, 2, 2)
    heatmap = ax.imshow(dic['kp_score_fg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_score_fg')

    ax = plt.subplot(2, 2, 3)
    heatmap = ax.imshow(dic['kp_prob_bg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_prob_bg')
    ax = plt.subplot(2, 2, 4)
    heatmap = ax.imshow(dic['kp_prob_fg'], aspect='equal', cmap=plt.cm.jet)
    plt.colorbar(heatmap)
    ax.set_title('kp_prob_fg')

    # plt.tight_layout()
    plt.close('all')

def visual_convs(im, blobs):
    dic = {}
    dic['conv3_3'] = np.mean(blobs['conv3_3'].data[0], 0)
    dic['conv4_3'] = np.mean(blobs['conv4_3'].data[0], 0)
    dic['conv5_3'] = np.mean(blobs['conv5_3'].data[0], 0)
    plt.figure(0)
    ax = plt.subplot(2, 2, 1)
    ax.imshow(im[:, :, ::-1], aspect='equal')
    ax.set_title('img')
    ax = plt.subplot(2, 2, 2)
    heatmap = ax.imshow(dic['conv3_3'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv3_3')
    ax = plt.subplot(2, 2, 3)
    heatmap = ax.imshow(dic['conv4_3'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv4_3')
    ax = plt.subplot(2, 2, 4)
    heatmap = ax.imshow(dic['conv5_3'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv5_3')
    plt.close('all')

def visual_fpn_convs(im, blobs):
    dic = {}
    dic['conv3_3'] = np.mean(blobs['conv3_3'].data[0], 0)
    dic['conv4_3'] = np.mean(blobs['conv4_3'].data[0], 0)
    dic['conv5_1'] = np.mean(blobs['conv5_1'].data[0], 0)
    dic['conv5_2'] = np.mean(blobs['conv5_2'].data[0], 0)
    dic['conv5_3'] = np.mean(blobs['conv5_3'].data[0], 0)
    dic['conv4_1'] = np.mean(blobs['conv4_1'].data[0], 0)
    dic['conv4_2'] = np.mean(blobs['conv4_2'].data[0], 0)
    dic['conv3_2'] = np.mean(blobs['conv3_2'].data[0], 0)
    dic['p4'] = np.mean(blobs['p4'].data[0], 0)
    dic['p5'] = np.mean(blobs['p5'].data[0], 0)
    dic['p6'] = np.mean(blobs['p6'].data[0], 0)
    dic['rpn_cls_score_4'] = np.mean(blobs['rpn_cls_score_4'].data[0], 0)
    dic['rpn_cls_score_5'] = np.mean(blobs['rpn_cls_score_5'].data[0], 0)
    dic['rpn_cls_score_6'] = np.mean(blobs['rpn_cls_score_6'].data[0], 0)
    plt.figure(0)
    ax = plt.subplot(3, 3, 1)
    ax.imshow(im[:, :, ::-1], aspect='equal')
    ax.set_title('img')
    ax = plt.subplot(3, 3, 2)
    heatmap = ax.imshow(dic['conv3_2'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv3_2')
    ax = plt.subplot(3, 3, 3)
    heatmap = ax.imshow(dic['conv3_3'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv3_3')
    ax = plt.subplot(3, 3, 4)
    heatmap = ax.imshow(dic['conv4_1'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv4_1')
    ax = plt.subplot(3, 3, 5)
    heatmap = ax.imshow(dic['conv4_2'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv4_2')
    ax = plt.subplot(3, 3, 6)
    heatmap = ax.imshow(dic['conv4_3'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv4_3')
    ax = plt.subplot(3, 3, 7)
    heatmap = ax.imshow(dic['conv5_1'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv5_1')
    ax = plt.subplot(3, 3, 8)
    heatmap = ax.imshow(dic['conv5_2'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv5_2')
    ax = plt.subplot(3, 3, 9)
    heatmap = ax.imshow(dic['conv5_3'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('conv5_3')

    plt.figure(1)
    ax = plt.subplot(2, 3, 1)
    heatmap = ax.imshow(dic['p4'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('p4')
    ax = plt.subplot(2, 3, 2)
    heatmap = ax.imshow(dic['p5'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('p5')
    ax = plt.subplot(2, 3, 3)
    heatmap = ax.imshow(dic['p6'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('p6')
    ax = plt.subplot(2, 3, 4)
    heatmap = ax.imshow(dic['rpn_cls_score_4'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('rpn_cls_score_4')
    ax = plt.subplot(2, 3, 5)
    heatmap = ax.imshow(dic['rpn_cls_score_5'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('rpn_cls_score_5')
    ax = plt.subplot(2, 3, 6)
    heatmap = ax.imshow(dic['rpn_cls_score_6'], aspect='equal', cmap=plt.cm.jet)
    # plt.colorbar(heatmap)
    ax.set_title('rpn_cls_score_6')
    plt.close('all')


def visual_kp_boxes(im, boxes, scores, keyPoint_map, pred_keyPoints, offset_scale, subplot_x=2, subplot_y=3, threshold=0.8):
    plt.figure(0)
    boxes = boxes[np.where(scores > threshold)[0]]
    pred_keyPoints = pred_keyPoints[np.where(scores > threshold)[0]]
    boxes_t = boxes.copy()
    boxes_o_w = boxes[:, 2] - boxes[:, 0]
    boxes_o_h = boxes[:, 3] - boxes[:, 1]
    boxes_t_w = boxes_o_w * offset_scale
    boxes_t_h = boxes_o_h * offset_scale
    for i, wd in enumerate(boxes_t_w > boxes_t_h):
        if wd:
            boxes_t_h[i] = boxes_t_w[i]
        else:
            boxes_t_w[i] = boxes_t_h[i]
    boxes_t[:, 0] = boxes[:, 0] - (boxes_t_w - boxes_o_w) / 2
    boxes_t[:, 2] = boxes[:, 2] + (boxes_t_w - boxes_o_w) / 2
    boxes_t[:, 1] = boxes[:, 1] - (boxes_t_h - boxes_o_h) / 2
    boxes_t[:, 3] = boxes[:, 3] + (boxes_t_h - boxes_o_h) / 2
    num = len(boxes_t)
    for i in range(num):
        ax = plt.subplot(subplot_x, subplot_y, i+1)
        kp_point_x = pred_keyPoints[i].reshape(-1, 2)[:, 0] - boxes_t[i][0]
        kp_point_y = pred_keyPoints[i].reshape(-1, 2)[:, 1] - boxes_t[i][1]
        plt.plot(kp_point_x, kp_point_y, 'go', ms=3.5, alpha=1, c='r', marker='o')
        img = im[boxes_t[i][1]:boxes_t[i][3], boxes_t[i][0]:boxes_t[i][2], :]
        ax.imshow(img[:, :, ::-1], aspect='equal')

    plt.figure(1)
    keyPoint_map = keyPoint_map[np.where(scores > threshold)[0]]
    for i in range(num):
        ax = plt.subplot(subplot_x, subplot_y, i+1)
        kp_map = np.sum(keyPoint_map[i][1:], 0)
        ax.imshow(kp_map, aspect='equal', cmap=plt.cm.jet)
    plt.close('all')

def tight_imshow(im, fig, ax):
    # fig, ax = plt.subplots()
    # im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width, channels = im.shape
    fig.set_size_inches(width / 100.0, height / 100.0)  # / 3.0
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

def visual_demos(im, boxes, scores, boxes_rpn, scores_rpn, pred_keyPoints,
                 threshold=0.8, vis_ori=0, vis_box=1, vis_rpnBox=0, vis_kp=1):
    if vis_ori:
        f = plt.figure(0)
        f.add_subplot(221)
        plt.imshow(im[:, :, ::-1], aspect='equal')
        subplot = f.add_subplot(222)
        plt.imshow(im[:, :, ::-1], aspect='equal')
        boxes_rpn = boxes_rpn[np.where(scores_rpn > threshold+0.18)[0]]
        for box_rpn in boxes_rpn:
            draw_bbox(subplot, box_rpn)
        subplot = f.add_subplot(223)
        plt.imshow(im[:, :, ::-1], aspect='equal')
        boxes = boxes[np.where(scores > threshold)[0]]
        for box in boxes:
            draw_bbox(subplot, box)
        f.add_subplot(224)
        plt.imshow(im[:, :, ::-1], aspect='equal')
        pred_keyPoints = pred_keyPoints[np.where(scores > threshold)[0]]
        for i in range(len(pred_keyPoints)):
            kp_point_x = pred_keyPoints[i].reshape(-1, 2)[:, 0]
            kp_point_y = pred_keyPoints[i].reshape(-1, 2)[:, 1]
            plt.plot(kp_point_x, kp_point_y, 'go', ms=3.5, alpha=1, c='r', marker='o')
    else:
        imname = cfg.TRAIN.VISUAL_ANCHORS_IMG.split('/')[-1]
        imdir = "/home/sean/workplace/221/py-R-FCN-test/data/demo/temp/%s" % cfg.TRAIN.VISUAL_ANCHORS_IMG.split('/')[-3]
        mk_dir(imdir, 0)
        imfile = os.path.join(imdir, imname)
        f = plt.figure(0)
        ax = f.add_subplot(111)
        tight_imshow(im[:, :, ::-1], f, ax)
        ax.set_axis_off()

        if vis_rpnBox == 1:
            boxes_rpn = boxes_rpn[np.where(scores_rpn > threshold + 0.18)[0]]
            for box_rpn in boxes_rpn:
                draw_bbox(ax, box_rpn)
        if vis_box == 1:
            boxes = boxes[np.where(scores > threshold)[0]]
            for box in boxes:
                draw_bbox(ax, box)
        if vis_kp == 1:
            pred_keyPoints = pred_keyPoints[np.where(scores > threshold)[0]]
            for i in range(len(pred_keyPoints)):
                kp_point_x = pred_keyPoints[i].reshape(-1, 2)[:, 0]
                kp_point_y = pred_keyPoints[i].reshape(-1, 2)[:, 1]
                plt.plot(kp_point_x, kp_point_y, 'go', ms=3.5, alpha=1, c='r', marker='o')

        plt.savefig(imfile, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        # exit(1)
    plt.close('all')


def label2img(labels):
    R = range(0, 255, 3)
    G = range(255, 0, -3)
    B = range(0, 255, 3)
    h, w = labels.shape
    seg_im = np.zeros([h, w, 3], dtype=np.uint8)
    seg_im = seg_im.reshape(h * w, 3)
    for i, v in enumerate(labels.ravel()):
        if v == 0:
            seg_im[i] = [255, 255, 255]
        else:
            seg_im[i] = [R[v - 1], G[v - 1], B[v - 1]]
    seg_im = seg_im.reshape([h, w, 3])
    return seg_im

def MaxMinNormalization(x):
    if len(x.shape) == 2:
        h = x.shape[0]
        w = x.shape[1]
        x = x.reshape(h * w, 1)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = x.reshape(h, w)
    else:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def im_detect_facePlus_v1(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()
    print ('forward took {:.3f}s').format(timer.total_time)
    if cfg.TEST.VISUAL_FEATURE:
        vis_square(net.blobs['conv1_1'].data[0])
        vis_square(net.blobs['conv2_1'].data[0, :36])
        vis_square(net.blobs['conv3_1'].data[0, :36])

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                rois_from_p4_3 = net.blobs['rois_from_p4_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
            else:
                rois_from_p4_3 = net.blobs['rois_from_p4_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
                rois_from_p3_3 = net.blobs['rois_from_p3_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p3_3 = rois_from_p3_3[:, 1:5] / im_scales[0]

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                rois_from_p4_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
            else:
                rois_from_p4_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
                rois_from_p3_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p3_3 = rois_from_p3_3[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores = np.vstack((scores, scores_from_p4_3))
            else:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores_from_p3_3 = blobs_out['cls_prob_from_p3_3']
                scores = np.vstack((scores, scores_from_p4_3, scores_from_p3_3))

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores = np.vstack((scores, scores_from_p4_3))
            else:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores_from_p3_3 = blobs_out['cls_prob_from_p3_3']
                scores = np.vstack((scores, scores_from_p4_3, scores_from_p3_3))

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3))
            else:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                box_deltas_from_p3_3 = blobs_out['bbox_pred_from_p3_3']
                pred_boxes_from_p3_3 = bbox_transform_inv(boxes_from_p3_3, box_deltas_from_p3_3)
                pred_boxes_from_p3_3 = clip_boxes(pred_boxes_from_p3_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3, pred_boxes_from_p3_3))

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3))
            else:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                box_deltas_from_p3_3 = blobs_out['bbox_pred_from_p3_3']
                pred_boxes_from_p3_3 = bbox_transform_inv(boxes_from_p3_3, box_deltas_from_p3_3)
                pred_boxes_from_p3_3 = clip_boxes(pred_boxes_from_p3_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3, pred_boxes_from_p3_3))

    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    # add keyPoints out keyPoint_pred
    if cfg.TRAIN.RPN_KP_REGRESSION:
        rois_keyPoints = blobs_out['rois_keyPoints']
        # unscale back to raw image space
        rois_keyPoints = rois_keyPoints[:, 1:] / im_scales[0]
        keyPoint_deltas = blobs_out['keyPoint_pred']
        pred_keyPoints = keyPoints_transform_inv(boxes, keyPoint_deltas, rois_keyPoints, 'elewise')
    else:
        keyPoint_deltas = blobs_out['keyPoint_pred']
        pred_keyPoints = keyPoints_transform_inv(boxes, keyPoint_deltas)

    # add other attributes out age_prob gender_prob ethnicity_prob
    pred_age_prob = blobs_out['age_prob']
    pred_gender_prob = blobs_out['gender_prob']
    pred_ethnicity_prob = blobs_out['ethnicity_prob']

    return scores, pred_boxes, pred_keyPoints, pred_age_prob, pred_gender_prob, pred_ethnicity_prob


def im_detect_facePlus_v1_by_rois(net, im, boxes=None, rois_layer = 'rois'):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs[rois_layer] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs[rois_layer] = blobs[rois_layer][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs[rois_layer].reshape(*(blobs[rois_layer].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs[rois_layer] = blobs[rois_layer].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()
    print ('forward took {:.3f}s').format(timer.total_time)
    if cfg.TEST.VISUAL_FEATURE:
        vis_square(net.blobs['conv1_1'].data[0])
        vis_square(net.blobs['conv2_1'].data[0, :36])
        vis_square(net.blobs['conv3_1'].data[0, :36])

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        # unscale back to raw image space
        boxes = net.blobs[rois_layer].data[:, 1:].copy() / im_scales[0]
        if rois_layer == 'rois':
            scores = net.blobs['rpn_scores'].data.copy()
        elif rois_layer == 'fc_rois':
            scores = net.blobs['fc_scores'].data.copy()

    # add keyPoints out keyPoint_pred
    if cfg.TRAIN.RPN_KP_REGRESSION:
        rois_keyPoints = blobs_out['rois_keyPoints']
        # unscale back to raw image space
        rois_keyPoints = rois_keyPoints[:, 1:] / im_scales[0]
        keyPoint_deltas = blobs_out['keyPoint_pred']
        pred_keyPoints = keyPoints_transform_inv(boxes, keyPoint_deltas, rois_keyPoints, 'elewise')
    else:
        keyPoint_deltas = blobs_out['keyPoint_pred']

        keyPoint_num = 136 if cfg.TRAIN.KP == 7 else cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
        keyPoint_deltas = keyPoint_deltas[:, keyPoint_num:]

        pred_keyPoints = keyPoints_transform_inv(boxes, keyPoint_deltas)

    # add other attributes out age_prob gender_prob ethnicity_prob
    pred_age_prob = blobs_out['age_prob']
    pred_gender_prob = blobs_out['gender_prob']
    pred_ethnicity_prob = blobs_out['ethnicity_prob']

    return scores, boxes, pred_keyPoints, pred_age_prob, pred_gender_prob, pred_ethnicity_prob


def im_detect_morph(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()
    print ('forward took {:.3f}s').format(timer.total_time)
    if cfg.TEST.VISUAL_FEATURE:
        vis_square(net.blobs['conv1_1'].data[0])
        vis_square(net.blobs['conv2_1'].data[0, :36])
        vis_square(net.blobs['conv3_1'].data[0, :36])

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                rois_from_p4_3 = net.blobs['rois_from_p4_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
            else:
                rois_from_p4_3 = net.blobs['rois_from_p4_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
                rois_from_p3_3 = net.blobs['rois_from_p3_3'].data.copy()
                # unscale back to raw image space
                boxes_from_p3_3 = rois_from_p3_3[:, 1:5] / im_scales[0]

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                rois_from_p4_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
            else:
                rois_from_p4_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p4_3 = rois_from_p4_3[:, 1:5] / im_scales[0]
                rois_from_p3_3 = net.blobs['rois'].data.copy()
                # unscale back to raw image space
                boxes_from_p3_3 = rois_from_p3_3[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores = np.vstack((scores, scores_from_p4_3))
            else:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores_from_p3_3 = blobs_out['cls_prob_from_p3_3']
                scores = np.vstack((scores, scores_from_p4_3, scores_from_p3_3))

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores = np.vstack((scores, scores_from_p4_3))
            else:
                scores_from_p4_3 = blobs_out['cls_prob_from_p4_3']
                scores_from_p3_3 = blobs_out['cls_prob_from_p3_3']
                scores = np.vstack((scores, scores_from_p4_3, scores_from_p3_3))

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
        if cfg.PYRAMID_MORE and not cfg.PYRAMID_ONEFC:
            if len(cfg.PYRAMID_MORE_ANCHORS) == 2:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3))
            else:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                box_deltas_from_p3_3 = blobs_out['bbox_pred_from_p3_3']
                pred_boxes_from_p3_3 = bbox_transform_inv(boxes_from_p3_3, box_deltas_from_p3_3)
                pred_boxes_from_p3_3 = clip_boxes(pred_boxes_from_p3_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3, pred_boxes_from_p3_3))

        if cfg.MIXED_PYRAMID_MORE:
            if cfg.MIXED_PYRAMID_NUM == 2:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3))
            else:
                box_deltas_from_p4_3 = blobs_out['bbox_pred_from_p4_3']
                pred_boxes_from_p4_3 = bbox_transform_inv(boxes_from_p4_3, box_deltas_from_p4_3)
                pred_boxes_from_p4_3 = clip_boxes(pred_boxes_from_p4_3, im.shape)
                box_deltas_from_p3_3 = blobs_out['bbox_pred_from_p3_3']
                pred_boxes_from_p3_3 = bbox_transform_inv(boxes_from_p3_3, box_deltas_from_p3_3)
                pred_boxes_from_p3_3 = clip_boxes(pred_boxes_from_p3_3, im.shape)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_from_p4_3, pred_boxes_from_p3_3))

    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    # add other attributes out age_prob gender_prob ethnicity_prob
    pred_age_prob = blobs_out['age_prob']
    pred_gender_prob = blobs_out['gender_prob']
    pred_ethnicity_prob = blobs_out['ethnicity_prob']

    return scores, pred_boxes, pred_age_prob, pred_gender_prob, pred_ethnicity_prob


def im_detect_morph_by_rois(net, im, boxes=None, rois_layer = 'rois'):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs[rois_layer] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs[rois_layer] = blobs[rois_layer][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs[rois_layer].reshape(*(blobs[rois_layer].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs[rois_layer] = blobs[rois_layer].astype(np.float32, copy=False)
    timer = Timer()
    timer.tic()
    blobs_out = net.forward(**forward_kwargs)
    timer.toc()
    print ('forward took {:.3f}s').format(timer.total_time)

    if cfg.TEST.VISUAL_FEATURE:
        vis_square(net.blobs['conv1_1'].data[0])
        vis_square(net.blobs['conv2_1'].data[0, :36])
        vis_square(net.blobs['conv3_1'].data[0, :36])

    if cfg.TEST.PRINT_FUSE_WEIGHT:
        print_fuse_weight(net)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        # unscale back to raw image space
        boxes = net.blobs[rois_layer].data[:, 1:].copy() / im_scales[0]
        if rois_layer == 'rois':
            scores = net.blobs['rpn_scores'].data.copy()
        elif rois_layer == 'fc_rois':
            scores = net.blobs['fc_scores'].data.copy()

    # add other attributes out age_prob gender_prob ethnicity_prob
    pred_age_prob = blobs_out['age_prob']
    pred_gender_prob = blobs_out['gender_prob']
    pred_ethnicity_prob = blobs_out['ethnicity_prob']

    return scores, boxes, pred_age_prob, pred_gender_prob, pred_ethnicity_prob


def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(net, imdb, max_per_image=400, thresh=-np.inf, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            if cfg.TEST.AGNOSTIC:
                cls_boxes = boxes[inds, 4:8]
            else:
                cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)


def MaxMinNormalization(x):
    if len(x.shape) == 2:
        h = x.shape[0]
        w = x.shape[1]
        x = x.reshape(h * w, 1)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = x.reshape(h, w)
    else:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def Softmax(x):
    import math
    x = math.e**x / sum(math.e**x)
    return x

def print_fuse_weight(net):
    fuse_layer_str = cfg.TEST.PRINT_FUSE_LAYER
    x = net.params[fuse_layer_str][0].data[0].ravel()
    y = Softmax(x)
    weights = []
    for channel in cfg.TEST.PRINT_FUSE_LAYER_CHANNELS:
        weight = sum(y[0:channel])  # / float(channel)
        weights.append(weight)
        y = y[channel:]
        print '%d channels weight is %f' % (channel, weight)