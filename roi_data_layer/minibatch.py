# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, prep_im_for_blob_ron, im_list_to_blob
from fast_rcnn.nms_wrapper import nms
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from fast_rcnn.bbox_transform import clip_boxes

class customError(StandardError):
    pass

def attribute_map(attrs, attrName):
    if attrName == 'gt_ages':
        attrs_str = map(lambda k: '%d years old' %k if k!=-1 else '', attrs)
    elif attrName == 'gt_genders':
        gender_map = {0:'Male',1:'Female',-1:""}
        attrs_str = map(lambda k: gender_map[k], attrs)
    elif attrName == 'gt_ethnicity':
        ethnicity_map = {0:'White',2:'Asian',1:'Black',-1:""}
        attrs_str = map(lambda k: ethnicity_map[k], attrs)
    return attrs_str

def visual_attribute(im_path, scale, gt_boxes, blobs, Flipped=True, fontsize=12, fontcolor='green'):
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    im = cv2.imread(im_path)
    if Flipped:
        im = im[:, ::-1, :]
    ax.imshow(im[:, :, ::-1], aspect='equal')
    ax = plt.subplot(1, 2, 2)
    # if cfg.ENABLE_RON:
    #     im = cv2.resize(im, None, None, fx=scale[0], fy=scale[1],
    #                     interpolation=cv2.INTER_LINEAR)
    # else:
    im = cv2.resize(im, None, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR)
    ax.imshow(im[:, :, ::-1], aspect='equal')

    label_num = len(cfg.TRAIN.ATTRIBUTES) -1
    labels = []
    if cfg.TRAIN.USE_ATTRIBUTE:
        for attr in cfg.TRAIN.ATTRIBUTES:
            attr_name = attr.keys()[0]
            if attr_name != "gt_keyPoints":
                labels.append(attribute_map(blobs[attr_name], attr_name))
            else:
                gt_keyPoints = blobs['gt_keyPoints']
                gt_keyPoints = gt_keyPoints.reshape((-1, 2))
                plt.plot(gt_keyPoints[:, 0], gt_keyPoints[:, 1], 'go', ms=1.5, alpha=1)

    for index, gt_boxe in enumerate(gt_boxes):
        gt_boxe_w = gt_boxe[2] - gt_boxe[0]
        gt_boxe_h = gt_boxe[3] - gt_boxe[1]
        rec = Rectangle((gt_boxe[0], gt_boxe[1]), width=gt_boxe_w, height=gt_boxe_h,
                        ec='r', fill=False, lw=1.5)
        ax.add_patch(rec)
        ax.text(gt_boxe[0] + fontsize*2, gt_boxe[3] + fontsize*6*label_num,
                '\n'.join([str(label[index]) for label in labels]),
                color=fontcolor,
                fontsize=fontsize,
                alpha=1,
                bbox={'facecolor': 'white', 'alpha': 0.5})

    plt.close('all')

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    if roidb[0]['flipped'] == True:
        cfg.TRAIN.VISUAL_ANCHORS_IMG_Flipped = True
    else:
        cfg.TRAIN.VISUAL_ANCHORS_IMG_Flipped = False
    cfg.TRAIN.VISUAL_ANCHORS_IMG = roidb[0]['image']

    num_images = len(roidb)
    num_reg_class = 2 if cfg.TRAIN.AGNOSTIC else num_classes
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0) or (cfg.TRAIN.BATCH_SIZE == -1), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = np.inf if cfg.TRAIN.BATCH_SIZE == -1 else cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)) if rois_per_image != float('inf') else np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
    # fix TypeError: 'numpy.float64' object cannot be interpreted as an index

    # Get the input image blob, formatted for caffe
    # (include mean subtract, scale, BGR order and reshape channel)
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    cfg.TRAIN.VISUAL_ANCHORS_IMG_SCALE = im_scales[0]

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        # if cfg.ENABLE_RON:
        #     boxes = roidb[0]['boxes'][gt_inds, :].copy()
        #     boxes[:, 0::2] = boxes[:, 0::2] * im_scales[0][0]
        #     boxes[:, 1::2] = boxes[:, 1::2] * im_scales[0][1]
        #     gt_boxes[:, 0:4] = boxes
        #     # not need to save scale if ENABLE_RON
        #     blobs['im_info'] = np.array(
        #         [[im_blob.shape[2], im_blob.shape[3], -1]],
        #         dtype=np.float32)
        # else:
        if cfg.TRANSFORM_KP_TO_BOX:
            gt_boxes[:, 0:4] = transform_kp_to_box(roidb[0]['gt_keyPoints'][gt_inds, :],
                                                   roidb[0]['boxes'][gt_inds, :],
                                                   roidb[0]['image']) * im_scales[0]
        else:
            gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]

        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes

        if cfg.TRAIN.USE_ATTRIBUTE:
            # if cfg.TRAIN.KP == 6:
            for attr in cfg.TRAIN.ATTRIBUTES:
                attributeName = attr.keys()[0]
                attributeNum = attr.values()[0]
                blobs[attributeName] = np.empty((len(gt_inds), attributeNum), dtype=np.float32)
                blobs[attributeName].fill(-1)
                if attributeName in roidb[0].keys():
                    if attributeName == 'gt_keyPoints':
                        # if cfg.ENABLE_RON:
                        #     gt_kp = roidb[0][attributeName][gt_inds, :].copy()
                        #     gt_kp[:, 0::2] = gt_kp[:, 0::2] * im_scales[0][0]
                        #     gt_kp[:, 1::2] = gt_kp[:, 1::2] * im_scales[0][1]
                        #     blobs[attributeName] = gt_kp
                        # else:
                        blobs[attributeName] = roidb[0][attributeName][gt_inds, :] * im_scales[0]
                    else:
                        blobs[attributeName] = roidb[0][attributeName][gt_inds]
            # else:
            #     for attributeName, attributeNum in cfg.TRAIN.ATTRIBUTES.iteritems():
            #         blobs[attributeName] = np.empty((len(gt_inds), attributeNum), dtype=np.float32)
            #         blobs[attributeName].fill(-1)
            #         if attributeName in roidb[0].keys():
            #             if attributeName == 'gt_keyPoints':
            #                 if cfg.TRAIN.KP == 1:
            #                     blobs[attributeName] = roidb[0][attributeName][gt_inds, :] * im_scales[0]  # missing label is -1
            #                 elif cfg.TRAIN.KP == 2:
            #                     blobs[attributeName] = roidb[0][attributeName][gt_inds, 38:] * im_scales[0]
            #                 elif cfg.TRAIN.KP == 3:
            #                     blobs[attributeName] = roidb[0][attributeName][gt_inds, 0:38] * im_scales[0]
            #                 elif cfg.TRAIN.KP == 4:
            #                     blobs[attributeName] = roidb[0][attributeName][gt_inds, 38:58] * im_scales[0]
            #                 elif cfg.TRAIN.KP == 5:
            #                     blobs[attributeName] = roidb[0][attributeName][gt_inds, :] * im_scales[0]
            #             else:
            #                 blobs[attributeName] = roidb[0][attributeName][gt_inds, :]

        if 0:
            im_path = roidb[0]['image']
            visual_attribute(im_path, im_scales[0], blobs['gt_boxes'], blobs, Flipped=roidb[0]['flipped'])

        # if cfg.TRAIN.USE_ATTRIBUTE:
        #     attributeNum = cfg.TRAIN.ATTRIBUTES.values()
        #     attributeSum = sum(attributeNum)
        #     blobs['attr_info'] = np.empty((len(gt_inds), attributeSum), dtype=np.float32)
        #     blobs['attr_info'].fill(-1)
        #     for attributeName, index in enumerate(cfg.TRAIN.ATTRIBUTES.keys()):
        #         if attributeName in roidb[0].keys():
        #             blobs['attr_info'][:, index:index+attributeNum[index]] = roidb[0][attributeName][gt_inds, :]
    else:  # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_reg_class), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

def transform_kp_to_box(gt_keyPoints, gt_boxes, im, fh=3):
    if cfg.FILTER_INVALID_BOX:
        kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints'] / 2
        gt_keyPoints = gt_keyPoints.reshape([-1, kp_num, 2])
        x1 = np.min(gt_keyPoints[:, :, 0], 1).reshape(-1, 1)
        y1 = np.min(gt_keyPoints[:, :, 1], 1).reshape(-1, 1)
        x2 = np.max(gt_keyPoints[:, :, 0], 1).reshape(-1, 1)
        y2 = np.max(gt_keyPoints[:, :, 1], 1).reshape(-1, 1)
        if cfg.WIDER_FACE_STYLE == 1:
            offset = (y2 - y1) / fh
            y1 = y1 - offset
        elif cfg.WIDER_FACE_STYLE == 2:
            if kp_num == 19:  # aflw-full
                y_offset = (y2 - y1) / fh
                y1 = y1 - y_offset
                x_offset = (gt_keyPoints[:, 1, 0] - gt_keyPoints[:, 0, 0])  # 2, 1
                x1 = x1 - x_offset
                x_offset = (gt_keyPoints[:, 5, 0] - gt_keyPoints[:, 4, 0])  # 5, 4
                x2 = x2 + x_offset
            elif kp_num == 29:  # cofw
                y_offset = (y2 - y1) / fh
                y1 = y1 - y_offset
                x_offset = (gt_keyPoints[:, 4, 0] - gt_keyPoints[:, 0, 0])
                x1 = x1 - x_offset
                x_offset = (gt_keyPoints[:, 1, 0] - gt_keyPoints[:, 6, 0])
                x2 = x2 + x_offset
        boxes = np.hstack([x1, y1, x2, y2])
    else:
        boxes = np.zeros([gt_keyPoints.shape[0], 4])
        for i, gt_keyPoint in enumerate(gt_keyPoints):
            if sum(gt_keyPoint) != 0:
                gt_keyPoint = gt_keyPoint.reshape([kp_num, 2])
                x1 = np.min(gt_keyPoint[:, 0])
                y1 = np.min(gt_keyPoint[:, 1])
                x2 = np.max(gt_keyPoint[:, 0])
                y2 = np.max(gt_keyPoint[:, 1])
                if cfg.WIDER_FACE_STYLE:
                    offset = (y2 - y1) / fh
                    y1 = y1 - offset
                boxes[i] = [x1, y1, x2, y2]
            else:
                boxes[i] = gt_boxes[i]
    if cfg.CLIP_BOXES:
        im_shape = cv2.imread(im).shape[0:2]
        boxes = clip_boxes(boxes, im_shape)
    return boxes

def get_ohem_minibatch_by_loss(loss, rois, labels, bbox_targets=None,
                       bbox_inside_weights=None, bbox_outside_weights=None):
    """Given rois and their loss, construct a minibatch using OHEM."""
    loss = np.array(loss)

    if cfg.TRAIN.OHEM_USE_NMS:
        # # Do NMS using loss for de-dup and diversity
        # keep_inds = []
        # nms_thresh = cfg.TRAIN.OHEM_NMS_THRESH
        # source_img_ids = [roi[0] for roi in rois]
        # for img_id in np.unique(source_img_ids):
        #     for label in np.unique(labels):
        #         sel_indx = np.where(np.logical_and(labels == label, \
        #                                            source_img_ids == img_id))[0]
        #         if not len(sel_indx):
        #             continue
        #         boxes = np.concatenate((rois[sel_indx, 1:],
        #                                 loss[sel_indx][:, np.newaxis]), axis=1).astype(np.float32)
        #         keep_inds.extend(sel_indx[nms(boxes, nms_thresh)])

        # Do NMS using loss for de-dup and diversity -m2
        # keep_inds = []
        # nms_thresh = cfg.TRAIN.OHEM_NMS_THRESH
        # for label in np.unique(labels):
        #     sel_indx = np.where(labels == label)[0]
        #     boxes = np.concatenate((rois[sel_indx, 1:],
        #                             loss[sel_indx][:, np.newaxis]), axis=1).astype(np.float32)
        #     keep_inds.extend(sel_indx[nms(box0es, nms_thresh)])

        keep_inds = nms(np.hstack((rois, loss)), cfg.TRAIN.OHEM_NMS_THRESH)
        # get labels filtered by nms
        fg_inds = np.where(labels[keep_inds] == 1)[0]
        bg_inds = np.where(labels[keep_inds] == 0)[0]

        hard_keep_inds = []
        num_fg = len(fg_inds)

        KEEP_BATCH_SIZE = True
        if num_fg >= cfg.TRAIN.BATCH_SIZE / 2:
            if KEEP_BATCH_SIZE:
                #half for top-64 fg, another half for top-64 bg
                fg_loss = loss[keep_inds][fg_inds]
                bg_loss = loss[keep_inds][bg_inds]
                hard_keep_inds.extend(fg_inds[select_topn_hard(fg_loss, cfg.TRAIN.BATCH_SIZE / 2)])
                hard_keep_inds.extend(bg_inds[select_topn_hard(bg_loss, cfg.TRAIN.BATCH_SIZE / 2)])
            else:
                # just append same amount of bg(highest loss)
                hard_keep_inds.extend(fg_inds)  # add fgs
                bg_loss = loss[keep_inds][bg_inds]
                hard_keep_inds.extend(bg_inds[select_topn_hard(bg_loss, num_fg)])
        else:  # add highest loss bg to make batch becomes 128
            hard_keep_inds.extend(fg_inds)  # add fgs
            num_bg = cfg.TRAIN.BATCH_SIZE - num_fg
            bg_loss = loss[keep_inds][bg_inds]
            hard_keep_inds.extend(bg_inds[select_topn_hard(bg_loss, num_bg)])

        # hard_keep_inds = select_hard_examples(loss[keep_inds])
        hard_inds = np.array(keep_inds)[hard_keep_inds]
    else:
        hard_inds = select_hard_examples(loss)

    blobs = {'rois_hard': rois[hard_inds, :].copy(),
             'labels_hard': labels[hard_inds].copy()}
    if bbox_targets is not None:
        assert cfg.TRAIN.BBOX_REG
        blobs['bbox_targets_hard'] = bbox_targets[hard_inds, :].copy()
        blobs['bbox_inside_weights_hard'] = bbox_inside_weights[hard_inds, :].copy()
        blobs['bbox_outside_weights_hard'] = bbox_outside_weights[hard_inds, :].copy()

    return blobs

def get_ohem_minibatch_by_score(fg_scores, rois, labels, bbox_targets=None,
                       bbox_inside_weights=None, bbox_outside_weights=None):

    # mining fg of which score is larger than threshold
    rois_per_image = cfg.TRAIN.BATCH_SIZE
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    if cfg.TRAIN.FC_OHEM_MINING_FG:
        fg_inds = np.where(labels == 1)[0]
        valid_fg_scores = fg_scores[fg_inds]
        if cfg.TRAIN.FC_OHEM_MINING_BALANCE == 1:
            fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
            sort_valid_fg_scores_inds = np.argsort(valid_fg_scores)
            sort_fg_inds = fg_inds[sort_valid_fg_scores_inds]
            valid_fg_inds = sort_fg_inds[:fg_rois_per_this_image]
        elif cfg.TRAIN.FC_OHEM_MINING_BALANCE == 2:
            valid_fg_inds = fg_inds
        else:
            valid_fg_scores_inds = np.where(valid_fg_scores < cfg.TRAIN.FC_OHEM_MINING_FG_threshold)[0]
            valid_fg_inds = fg_inds[valid_fg_scores_inds]
    else:
        fg_inds = np.where(labels == 1)[0]
        valid_fg_inds = fg_inds

    if cfg.TRAIN.FC_OHEM_MINING_BG:
        bg_inds = np.where(labels == 0)[0]
        valid_bg_scores = fg_scores[bg_inds]
        if cfg.TRAIN.FC_OHEM_MINING_BALANCE == 1:
            bg_num = rois_per_image - fg_rois_per_this_image
            sort_valid_bg_scores_inds = np.argsort(-1 * valid_bg_scores)
            sort_bg_inds = bg_inds[sort_valid_bg_scores_inds]
            valid_bg_inds = sort_bg_inds[:bg_num]
        elif cfg.TRAIN.FC_OHEM_MINING_BALANCE == 2:
            bg_num = len(fg_inds)
            if len(bg_inds) > bg_num:
                sort_valid_bg_scores_inds = np.argsort(-1 * valid_bg_scores)
                sort_bg_inds = bg_inds[sort_valid_bg_scores_inds]
                valid_bg_inds = sort_bg_inds[:bg_num]
            else:
                valid_bg_inds = bg_inds
        else:
            valid_bg_scores_inds = np.where(valid_bg_scores > cfg.TRAIN.FC_OHEM_MINING_BG_threshold)[0]
            valid_bg_inds = bg_inds[valid_bg_scores_inds]
    else:
        bg_inds = np.where(labels == 0)[0]
        valid_bg_inds = bg_inds

    hard_inds = np.hstack([valid_fg_inds, valid_bg_inds])
    blobs = {'rois_hard': rois[hard_inds, :].copy(),
             'labels_hard': labels[hard_inds].copy()}
    if bbox_targets is not None:
        assert cfg.TRAIN.BBOX_REG
        blobs['bbox_targets_hard'] = bbox_targets[hard_inds, :].copy()
        blobs['bbox_inside_weights_hard'] = bbox_inside_weights[hard_inds, :].copy()
        blobs['bbox_outside_weights_hard'] = bbox_outside_weights[hard_inds, :].copy()

    return blobs

def get_rohem_minibatch_by_score(fg_scores, labels, bbox_targets=None,
                       bbox_inside_weights=None, bbox_outside_weights=None):

    # mining fg of which score is larger than threshold
    rois_per_image = cfg.TRAIN.BATCH_SIZE
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    if cfg.TRAIN.FC_OHEM_MINING_FG:
        fg_inds = np.where(labels == 1)[0]
        valid_fg_scores = fg_scores[fg_inds]
        if cfg.TRAIN.FC_OHEM_MINING_BALANCE == 1:
            fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
            sort_valid_fg_scores_inds = np.argsort(valid_fg_scores)
            sort_fg_inds = fg_inds[sort_valid_fg_scores_inds]
            valid_fg_inds = sort_fg_inds[:fg_rois_per_this_image]
        elif cfg.TRAIN.FC_OHEM_MINING_BALANCE == 2:
            valid_fg_inds = fg_inds
        else:
            valid_fg_scores_inds = np.where(valid_fg_scores < cfg.TRAIN.FC_OHEM_MINING_FG_threshold)[0]
            valid_fg_inds = fg_inds[valid_fg_scores_inds]
    else:
        fg_inds = np.where(labels == 1)[0]
        valid_fg_inds = fg_inds

    if cfg.TRAIN.FC_OHEM_MINING_BG:
        bg_inds = np.where(labels == 0)[0]
        valid_bg_scores = fg_scores[bg_inds]
        if cfg.TRAIN.FC_OHEM_MINING_BALANCE == 1:
            bg_num = rois_per_image - fg_rois_per_this_image
            sort_valid_bg_scores_inds = np.argsort(-1 * valid_bg_scores)
            sort_bg_inds = bg_inds[sort_valid_bg_scores_inds]
            valid_bg_inds = sort_bg_inds[:bg_num]
        elif cfg.TRAIN.FC_OHEM_MINING_BALANCE == 2:
            bg_num = len(fg_inds)
            if len(bg_inds) > bg_num:
                sort_valid_bg_scores_inds = np.argsort(-1 * valid_bg_scores)
                sort_bg_inds = bg_inds[sort_valid_bg_scores_inds]
                valid_bg_inds = sort_bg_inds[:bg_num]
            else:
                valid_bg_inds = bg_inds
        else:
            valid_bg_scores_inds = np.where(valid_bg_scores > cfg.TRAIN.FC_OHEM_MINING_BG_threshold)[0]
            valid_bg_inds = bg_inds[valid_bg_scores_inds]
    else:
        bg_inds = np.where(labels == 0)[0]
        valid_bg_inds = bg_inds

    labels[:] = -1
    bbox_inside_weights[:] = np.zeros(bbox_targets.shape, dtype=np.float32)
    bbox_outside_weights[:] = np.zeros(bbox_targets.shape, dtype=np.float32)

    if len(valid_bg_inds) != 0:
        labels[valid_bg_inds] = 0
    if len(valid_fg_inds) != 0:
        labels[valid_fg_inds] = 1
        cls = 2
        start = 4 * (1 if cls > 0 else 0)
        end = start + 4
        bbox_inside_weights[valid_fg_inds, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
        bbox_outside_weights[valid_fg_inds, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

    blobs = {'labels_hard': labels.copy()}
    if bbox_targets is not None:
        assert cfg.TRAIN.BBOX_REG
        blobs['bbox_targets_hard'] = bbox_targets.copy()
        blobs['bbox_inside_weights_hard'] = bbox_inside_weights.copy()
        blobs['bbox_outside_weights_hard'] = bbox_outside_weights.copy()

    return blobs

def select_topn_hard(loss, n):
    sorted_inds = np.argsort(loss.ravel())[::-1]
    return sorted_inds[0:n]


def select_hard_examples_bak(loss):  # backup version
    """Select hard rois."""
    # Sort and select top hard examples.
    sorted_indices = np.argsort(loss)[::-1]
    hard_keep_inds = sorted_indices[0:np.minimum(len(loss), cfg.TRAIN.BATCH_SIZE)]
    # (explore more ways of selecting examples in this function; e.g., sampling)
    return hard_keep_inds


def select_hard_examples(loss):  # backup version
    """Select hard rois."""
    # Sort and select top hard examples.
    sorted_indices = np.argsort(loss.ravel())[::-1]
    hard_keep_inds = sorted_indices[0:np.minimum(len(loss), cfg.TRAIN.BATCH_SIZE)]
    # (explore more ways of selecting examples in this function; e.g., sampling)
    return hard_keep_inds


def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])  # BGR format
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        if cfg.TRAIN.MULTI_SCALE:
            target_size = roidb[i]['scale']
        else:
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        # mean subtract and scale
        # if cfg.ENABLE_RON:
        #     im, im_scale = prep_im_for_blob_ron(im, cfg.PIXEL_MEANS, target_size)
        # else:
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images (change channel (0, 3, 1, 2))
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    num_reg_class = 2 if cfg.TRAIN.AGNOSTIC else num_classes
    bbox_targets = np.zeros((clss.size, 4 * num_reg_class), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]

    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = 4 * int(cls)  # fix slice indices must be integers or None or have an __index__ method
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
