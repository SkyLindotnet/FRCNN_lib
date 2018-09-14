# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_ohem_minibatch_by_loss, get_ohem_minibatch_by_score, get_rohem_minibatch_by_score
import numpy as np
import yaml
from multiprocessing import Process, Queue

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            if cfg.TRAIN.USE_FLIPPED:
                inds = np.reshape(inds, (-1, 2))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1,))
            else:
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = inds[row_perm]
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0


    def _shuffle_roidb_inds_heterology(self, heterologyStrList):
        """Randomly permute the training roidb."""
        heterologyListNum = len(heterologyStrList)
        heterology_i = self._cur % heterologyListNum
        if self._cur == 0:
            self.heterologyIndList = np.zeros(heterologyListNum, dtype=np.int)
            self.heterologyRoiList = self._get_heterology_list(heterologyStrList)

            heterologyRoi = self.heterologyRoiList[heterology_i]
            heterologyRoiInd = self.heterologyIndList[heterology_i]
            inds = heterologyRoi[heterologyRoiInd]
        else:
            heterologyRoi = self.heterologyRoiList[heterology_i]
            heterologyRoiInd = self.heterologyIndList[heterology_i]
            heterology_i_num = len(heterologyRoi)

            if heterologyRoiInd == heterology_i_num:
                self.heterologyRoiList[heterology_i] = np.random.permutation(heterologyRoi)
                self.heterologyIndList[heterology_i] = 0
                heterologyRoi = self.heterologyRoiList[heterology_i]
                heterologyRoiInd = self.heterologyIndList[heterology_i]
                inds = heterologyRoi[heterologyRoiInd]
            else:
                inds = heterologyRoi[heterologyRoiInd]

        self.heterologyIndList[heterology_i] = heterologyRoiInd + 1
        return inds

    def _get_heterology_list(self, listStr):
        heterology_list = [[] for str in listStr]
        inds = self._perm
        for ind in inds:
            roi = self._roidb[ind]
            roi_img = roi['image']
            for index, dbs in enumerate(listStr):
                for db in dbs:
                    if db in roi_img:
                        heterology_list[index].append(ind)
        return heterology_list

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch_inds_heterology(self):
        """Return the roidb indices for the next minibatch."""
        db_inds = self._shuffle_roidb_inds_heterology(cfg.TRAIN.shuffle_heterologyList)
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return [db_inds]

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            if cfg.TRAIN.shuffle_heterology:
                db_inds = self._get_next_minibatch_inds_heterology()  # get a index of sample from random list
            else:
                db_inds = self._get_next_minibatch_inds()  # get a index of sample from random list
            minibatch_db = [self._roidb[i] for i in db_inds]

            # print "Image: {}".format(minibatch_db[0]['image'])

            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()  # randomly arrange inds by grouping
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0

        if cfg.TRAIN.MULTI_SCALE:
            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
                cfg.TRAIN.MULTI_SCALE_MIN, cfg.TRAIN.MULTI_SCALE_MAX)
        else:
            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
                max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)

        if cfg.ENABLE_RON:
            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
                             cfg.TEST.RON_SCALES[0], cfg.TEST.RON_SCALES[0])

        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1

            if cfg.TRAIN.USE_ATTRIBUTE:
                # if cfg.TRAIN.KP == 6:
                for attr in cfg.TRAIN.ATTRIBUTES:
                    top[idx].reshape(1, attr.values()[0])
                    self._name_to_top_map[attr.keys()[0]] = idx
                    idx += 1
                # for attr_name, attr_num in cfg.TRAIN.ATTRIBUTES.iteritems():
                #     top[idx].reshape(1, attr_num)
                #     self._name_to_top_map[attr_name] = idx
                #     idx += 1

        else:  # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5, 1, 1)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1, 1, 1, 1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                num_reg_class = 2 if cfg.TRAIN.AGNOSTIC else self._num_classes
                top[idx].reshape(1, num_reg_class * 4, 1, 1)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, num_reg_class * 4, 1, 1)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, num_reg_class * 4, 1, 1)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()  # include gt_boxes, data(re_img), im_info(w,h,s)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            shape = blob.shape
            if len(shape) == 1:
                blob = blob.reshape(blob.shape[0], 1, 1, 1)
            if len(shape) == 2 and blob_name != 'im_info':
                blob = blob.reshape(blob.shape[0], blob.shape[1], 1, 1)
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class OHEMDataLayer(caffe.Layer):
    """Online Hard-example Mining Layer."""
    def setup(self, bottom, top):
        """Setup the OHEMDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_bottom_map = {
            'cls_prob_readonly': 0,
            'bbox_pred_readonly': 1,
            'rois': 2,
            'labels': 3}

        if cfg.TRAIN.BBOX_REG:
            self._name_to_bottom_map['bbox_targets'] = 4
            self._name_to_bottom_map['bbox_loss_weights'] = 5

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[idx].reshape(1, 5, 1, 1)  # (1, 5)
        self._name_to_top_map['rois_hard'] = idx
        idx += 1

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[idx].reshape(1, 1, 1, 1)  # (1)
        self._name_to_top_map['labels_hard'] = idx
        idx += 1

        if cfg.TRAIN.BBOX_REG:
            # bbox_targets blob: R bounding-box regression targets with 4
            # targets per class
            top[idx].reshape(1, self._num_classes * 4, 1, 1)  # (1, self._num_classes * 4)
            self._name_to_top_map['bbox_targets_hard'] = idx
            idx += 1

            # bbox_inside_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top[idx].reshape(1, self._num_classes * 4, 1, 1)  # (1, self._num_classes * 4)
            self._name_to_top_map['bbox_inside_weights_hard'] = idx
            idx += 1

            top[idx].reshape(1, self._num_classes * 4, 1, 1)  # (1, self._num_classes * 4)
            self._name_to_top_map['bbox_outside_weights_hard'] = idx
            idx += 1

        print 'OHEMDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

#  			  0 bottom: "cls_prob_readonly"
#  		      1 bottom: "bbox_pred_readonly"
#   		  2 bottom: "rois"	(2000,5)
#   		  3 bottom: "labels" (2000,) #fg1, bg0
#   		  4 bottom: "bbox_targets"
#   		  5 bottom: "bbox_inside_weights"
#   		  6 bottom: "bbox_outside_weights"

        cls_prob = bottom[0].data
        bbox_pred = bottom[1].data
        rois = bottom[2].data
        labels = bottom[3].data
        if cfg.TRAIN.BBOX_REG:
            bbox_target = bottom[4].data
            bbox_inside_weights = bottom[5].data
            bbox_outside_weights = bottom[6].data
        else:
            bbox_target = None
            bbox_inside_weights = None
            bbox_outside_weights = None

        # transform data into ohem format
        rois = rois.reshape([rois.shape[0], rois.shape[1]])
        labels = labels.reshape([labels.shape[0], labels.shape[1]]).astype(np.int)
        bbox_target = bbox_target.reshape([bbox_target.shape[0], bbox_target.shape[1]])
        bbox_inside_weights = bbox_inside_weights.reshape([bbox_inside_weights.shape[0], bbox_inside_weights.shape[1]])
        bbox_outside_weights = bbox_outside_weights.reshape([bbox_outside_weights.shape[0], bbox_outside_weights.shape[1]])

        if cfg.TRAIN.OHEM_BY_LOSS:
            flt_min = np.finfo(float).eps
            # classification loss
            loss = [-1 * np.log(max(x, flt_min))
                    for x in [cls_prob[i, label] for i, label in enumerate(labels)]]

            if cfg.TRAIN.BBOX_REG and cfg.TRAIN.OHEM_USE_BOX_LOSS:
                # bounding-box regression loss
                # d := w * (b0 - b1)
                # smoothL1(x) = 0.5 * x^2    if |x| < 1
                #               |x| - 0.5    otherwise
                def smoothL1(x):
                    if abs(x) < 1:
                        return 0.5 * x * x
                    else:
                        return abs(x) - 0.5

                bbox_loss = np.zeros(labels.shape[0])
                for i in np.where(labels > 0)[0]:
                    indices = np.where(bbox_inside_weights[i, :] != 0)[0]
                    bbox_loss[i] = sum(bbox_outside_weights[i, indices] *
                                       [smoothL1(x) for x in bbox_inside_weights[i, indices] *
                                        (bbox_pred[i, indices] - bbox_target[i, indices])])
                loss = np.array(loss).ravel()
                loss += bbox_loss
                loss = loss.reshape(loss.shape[0], 1)

            blobs = get_ohem_minibatch_by_loss(loss, rois, labels, bbox_target, \
                bbox_inside_weights, bbox_outside_weights)
        else:
            blobs = get_ohem_minibatch_by_score(cls_prob[:, 1], rois, labels, bbox_target, \
                bbox_inside_weights, bbox_outside_weights)

        len_hard = len(blobs['labels_hard'])
        if(len_hard != 128):
            print "after_ohem, label.len:",len_hard

        # for blob_name, blob in blobs.iteritems():
        #     top_ind = self._name_to_top_map[blob_name]
        #     # Reshape net's input blobs
        #     top[top_ind].reshape(*(blob.shape))
        #     # Copy data into net's input blobs
        #     top[top_ind].data[...] = blob.astype(np.float32, copy=False)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            shape = blob.shape
            if len(shape) == 1:
                blob = blob.reshape(blob.shape[0], 1, 1, 1)
            if len(shape) == 2:
                blob = blob.reshape(blob.shape[0], blob.shape[1], 1, 1)
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class ROHEMDataLayer(caffe.Layer):
    """Online Hard-example Mining Layer."""
    def setup(self, bottom, top):
        """Setup the OHEMDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[idx].reshape(1, 1, 1, 1)  # (1)
        self._name_to_top_map['labels_hard'] = idx
        idx += 1

        if cfg.TRAIN.BBOX_REG:
            # bbox_targets blob: R bounding-box regression targets with 4
            # targets per class
            top[idx].reshape(1, self._num_classes * 4, 1, 1)  # (1, self._num_classes * 4)
            self._name_to_top_map['bbox_targets_hard'] = idx
            idx += 1

            # bbox_inside_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top[idx].reshape(1, self._num_classes * 4, 1, 1)  # (1, self._num_classes * 4)
            self._name_to_top_map['bbox_inside_weights_hard'] = idx
            idx += 1

            top[idx].reshape(1, self._num_classes * 4, 1, 1)  # (1, self._num_classes * 4)
            self._name_to_top_map['bbox_outside_weights_hard'] = idx
            idx += 1

        print 'ROHEMDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

#  			  0 bottom: "cls_prob_readonly"
#  		      1 bottom: "bbox_pred_readonly"
#   		  2 bottom: "rois"	(2000,5)
#   		  3 bottom: "labels" (2000,) #fg1, bg0
#   		  4 bottom: "bbox_targets"
#   		  5 bottom: "bbox_inside_weights"
#   		  6 bottom: "bbox_outside_weights"

        cls_prob = bottom[0].data
        bbox_pred = bottom[1].data
        labels = bottom[2].data
        if cfg.TRAIN.BBOX_REG:
            bbox_target = bottom[3].data
            bbox_inside_weights = bottom[4].data
            bbox_outside_weights = bottom[5].data
        else:
            bbox_target = None
            bbox_inside_weights = None
            bbox_outside_weights = None

        # transform data into ohem format
        labels = labels.reshape([labels.shape[0], labels.shape[1]]).astype(np.int)
        bbox_target = bbox_target.reshape([bbox_target.shape[0], bbox_target.shape[1]])
        bbox_inside_weights = bbox_inside_weights.reshape([bbox_inside_weights.shape[0], bbox_inside_weights.shape[1]])
        bbox_outside_weights = bbox_outside_weights.reshape([bbox_outside_weights.shape[0], bbox_outside_weights.shape[1]])

        if cfg.TRAIN.OHEM_BY_LOSS:
            flt_min = np.finfo(float).eps
            # classification loss
            loss = [-1 * np.log(max(x, flt_min))
                    for x in [cls_prob[i, label] for i, label in enumerate(labels)]]

            if cfg.TRAIN.BBOX_REG and cfg.TRAIN.OHEM_USE_BOX_LOSS:
                # bounding-box regression loss
                # d := w * (b0 - b1)
                # smoothL1(x) = 0.5 * x^2    if |x| < 1
                #               |x| - 0.5    otherwise
                def smoothL1(x):
                    if abs(x) < 1:
                        return 0.5 * x * x
                    else:
                        return abs(x) - 0.5

                bbox_loss = np.zeros(labels.shape[0])
                for i in np.where(labels > 0)[0]:
                    indices = np.where(bbox_inside_weights[i, :] != 0)[0]
                    bbox_loss[i] = sum(bbox_outside_weights[i, indices] *
                                       [smoothL1(x) for x in bbox_inside_weights[i, indices] *
                                        (bbox_pred[i, indices] - bbox_target[i, indices])])
                loss = np.array(loss).ravel()
                loss += bbox_loss
                loss = loss.reshape(loss.shape[0], 1)

            # TODO get_rohem_minibatch_by_loss
            blobs = get_ohem_minibatch_by_loss(loss, rois, labels, bbox_target, \
                bbox_inside_weights, bbox_outside_weights)
        else:
            blobs = get_rohem_minibatch_by_score(cls_prob[:, 1], labels, bbox_target, \
                bbox_inside_weights, bbox_outside_weights)

        len_hard = len(blobs['labels_hard'])
        if(len_hard != 128):
            print "after_ohem, label.len:",len_hard

        # for blob_name, blob in blobs.iteritems():
        #     top_ind = self._name_to_top_map[blob_name]
        #     # Reshape net's input blobs
        #     top[top_ind].reshape(*(blob.shape))
        #     # Copy data into net's input blobs
        #     top[top_ind].data[...] = blob.astype(np.float32, copy=False)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            shape = blob.shape
            if len(shape) == 1:
                blob = blob.reshape(blob.shape[0], 1, 1, 1)
            if len(shape) == 2:
                blob = blob.reshape(blob.shape[0], blob.shape[1], 1, 1)
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)
