# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import matplotlib.pyplot as plt
from utils.face_test import generate_result_fddb, generate_result_wider_val, generate_result_voc_val, generate_result_gen_val
from utils.loss_tracker import run_plot_loss, run_plot_loss_1
from tempfile import NamedTemporaryFile
import math
import google.protobuf.text_format

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir, version,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.recordLossFile = None
        self.recordMetricFile = None
        self.max_iters = 0
        self.last_snapshot_iter = -1
        self.start_loss_iter = 1
        self.start_metric_iter = cfg.TRAIN.METRIC_ITERS

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        if cfg.TRAIN.WITH_SOLVERSTATE:
            self.solver_param = caffe_pb2.SolverParameter()
            with open(solver_prototxt, 'rt') as f:
                pb2.text_format.Merge(f.read(), self.solver_param)
            infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
            baseFileName = (self.solver_param.snapshot_prefix + infix + '_{:s}'.format(version))
            self.solver_param.snapshot_prefix = os.path.join(self.output_dir, baseFileName)
            with NamedTemporaryFile('w', delete=False) as f:
                f.write(pb2.text_format.MessageToString(self.solver_param))
            self.solver = caffe.SGDSolver(f.name)  # get_solver
        else:
            self.solver = caffe.SGDSolver(solver_prototxt)  # all layers from python module will be setup

        if pretrained_model is not None:
            if pretrained_model.endswith('.caffemodel'):
                print ('Loading pretrained model '
                       'weights from {:s}').format(pretrained_model)
                self.solver.net.copy_from(pretrained_model)
            elif pretrained_model.endswith('.solverstate'):
                print ('Loading solverstate'
                       ' from {:s}').format(pretrained_model)
                self.solver.restore(pretrained_model)
                self.last_snapshot_iter = self.solver.iter
                self.start_loss_iter = self.solver.iter + 1  # set start_iter to record loss
                if self.solver.iter + 1 > cfg.TRAIN.METRIC_ITERS:
                    offset_iters = cfg.TRAIN.METRIC_ITERS - (self.solver.iter + 1) % cfg.TRAIN.METRIC_ITERS
                    self.start_metric_iter = self.solver.iter + 1 + offset_iters

        if cfg.TRAIN.USE_SHARE_LAYER_COPY:
            # fc2_weight_copy
            net = self.solver.net
            for LAYERL_SUFFIX in cfg.TRAIN.TARGET_LAYERL_SUFFIX:
                lyr_name_fc = cfg.TRAIN.SOURCE_LAYERLIST
                lyr_name_fc2 = [i + LAYERL_SUFFIX for i in lyr_name_fc]

                for fc_name, fc2_name in zip(lyr_name_fc, lyr_name_fc2):
                    assert len(net.params[fc_name]) == len(net.params[fc2_name])
                    for i in range(len(net.params[fc_name])):
                        assert net.params[fc_name][i].data.shape == net.params[fc2_name][i].data.shape
                        # assert i <= 1
                        print 'copying {} "{}" --> "{}". shape:{} ...'.format(
                            'weights' if 0 else 'bias',
                            fc_name, fc2_name, net.params[fc_name][i].data.shape
                        ),
                        net.params[fc2_name][i].data.flat = net.params[fc_name][i].data.flat
                        print '[Done!]'

        if cfg.TRAIN.LOAD_MODEL_TWICE:
            # use post-process
            pretrained_model = cfg.TRAIN.LOAD_MODEL_TWICE_PATH
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        # record parameter of net
        # parameterPath = '/data6/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/wider_face/temp.txt'
        # with open(parameterPath, 'w') as f:
        #     for paramName in self.solver.net.params.keys():
        #         paramNum = len(self.solver.net.params[paramName])
        #         f.write('%s' % paramName)
        #         for i in range(paramNum):
        #             paramName_v = str(self.solver.net.params[paramName][i].data.ravel()[0:8])
        #             f.write(' %s' % paramName_v)
        #         f.write('\n')
        # exit(1)
        self.solver.net.layers[0].set_roidb(roidb)

    def recordLoss(self, version, stopIter, trainDB, startIter = 1, visualIter = 100):
        # self.solver.iter
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = ('Loss_' + self.solver_param.snapshot_prefix + infix +
                    '_{:s}'.format(version) + '.txt')
        filename = os.path.join(self.output_dir, filename)
        if self.solver.iter == startIter:
            self.recordLossFile = open(filename, 'w+')
        elif self.solver.iter == stopIter:
            self.recordLossFile.close()
            self.recordLossFile = None
        elif self.solver.iter > stopIter:
            return
        elif self.solver.iter < stopIter and self.recordLossFile is None:
            raise IOError(('recordLossFile is None before stop.\n'))

        if self.recordLossFile is not None:
            net = self.solver.net
            methodName = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '_{:s}'.format(version))
            if trainDB == 'wider_face' or trainDB == 'moon':
                if self.solver_param.snapshot_prefix == 'VGG16_rpn':
                    self.recordLossFile.write(methodName+' '+'rpn_cls_loss %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+
                                              'rpn_loss_bbox %.8f\n' % net.blobs['rpn_loss_bbox'].data)
                elif self.solver_param.snapshot_prefix == 'VGG16_rpn_v1':
                    self.recordLossFile.write(methodName+' '+'rpn_cls_loss %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+
                                              'rpn_loss_bbox %.8f' % net.blobs['rpn_loss_bbox'].data+
                                              ' '+'rpn_cls_loss_from_conv4_3 %.8f' %
                                              net.blobs['rpn_cls_loss_from_conv4_3'].data+
                                              ' '+'rpn_loss_bbox_from_conv4_3 %.8f\n' %
                                              net.blobs['rpn_loss_bbox_from_conv4_3'].data)
                elif self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_conv4':
                    allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + net.blobs['rpn_cls_loss'].data + net.blobs['rpn_loss_bbox'].data
                    self.recordLossFile.write(methodName+' '+'loss_bbox %.8f' %
                                              net.blobs['loss_bbox'].data+' '+'loss_cls %.8f' %
                                              net.blobs['loss_cls'].data+' '+'rpn_cls_loss_from_conv4_3 %.8f' %
                                              net.blobs['rpn_cls_loss_from_conv4_3'].data+' '+'rpn_loss_bbox_from_conv4_3 %.8f' %
                                              net.blobs['rpn_loss_bbox_from_conv4_3'].data+' '+'rpn_cls_loss %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+'rpn_loss_bbox %.8f' %
                                              net.blobs['rpn_loss_bbox'].data+' '+'all_loss %.8f\n' %
                                              allLoss)
                elif self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_pyramid_2' or \
                     self.solver_param.snapshot_prefix == 'VGG16_rfcn_end2end_with_pyramid_2':
                    allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + net.blobs['rpn_cls_loss'].data + net.blobs['rpn_loss_bbox'].data
                    allLoss_from_p4_3 = net.blobs['loss_bbox_from_p4_3'].data + net.blobs['loss_cls_from_p4_3'].data + net.blobs['rpn_cls_loss_from_p4_3'].data + net.blobs['rpn_loss_bbox_from_p4_3'].data

                    self.recordLossFile.write(methodName+' '+'loss_bbox %.8f' %
                                              net.blobs['loss_bbox'].data+' '+'loss_cls %.8f' %
                                              net.blobs['loss_cls'].data+' '+'loss_bbox_from_p4_3 %.8f' %
                                              net.blobs['loss_bbox_from_p4_3'].data+' '+'loss_cls_from_p4_3 %.8f' %
                                              net.blobs['loss_cls_from_p4_3'].data+' '+'rpn_cls_loss_from_p4_3 %.8f' %
                                              net.blobs['rpn_cls_loss_from_p4_3'].data+' '+'rpn_loss_bbox_from_p4_3 %.8f' %
                                              net.blobs['rpn_loss_bbox_from_p4_3'].data+' '+'rpn_cls_loss %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+'rpn_loss_bbox %.8f' %
                                              net.blobs['rpn_loss_bbox'].data+' '+'all_loss %.8f' %
                                              allLoss+' '+'all_loss_from_p4_3 %.8f\n' % allLoss_from_p4_3)
                elif self.solver_param.snapshot_prefix == 'VGG16_rfcn_end2end_with_mixed_pyramid_2':
                    allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + net.blobs['rpn_cls_loss'].data + net.blobs['rpn_loss_bbox'].data \
                              + net.blobs['loss_bbox_from_p4_3'].data + net.blobs['loss_cls_from_p4_3'].data

                    self.recordLossFile.write(methodName+' '+'loss_bbox %.8f' %
                                              net.blobs['loss_bbox'].data+' '+'loss_cls %.8f' %
                                              net.blobs['loss_cls'].data+' '+'loss_bbox_from_p4_3 %.8f' %
                                              net.blobs['loss_bbox_from_p4_3'].data+' '+'loss_cls_from_p4_3 %.8f' %
                                              net.blobs['loss_cls_from_p4_3'].data+' '+'rpn_cls_loss_from_p4_3 %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+'rpn_loss_bbox %.8f' %
                                              net.blobs['rpn_loss_bbox'].data+' '+'all_loss %.8f\n' %
                                              allLoss)
                elif self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_ohem' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_fuse' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_rfuse' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_multianchor' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_fuse_multianchor' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_pyramid_onefc' or \
                     self.solver_param.snapshot_prefix == 'VGG16_rfcn_end2end' or \
                     self.solver_param.snapshot_prefix == 'VGG16_rfcn_end2end_ohem' or \
                     self.solver_param.snapshot_prefix == 'VGG16_rfcn_end2end_with_multianchor' or \
                     self.solver_param.snapshot_prefix == 'VGG16_rfcn_end2end_with_fuse' or \
                     self.solver_param.snapshot_prefix == 'VGG16_rfcn_end2end_with_plus':
                    allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + net.blobs['rpn_cls_loss'].data + net.blobs['rpn_loss_bbox'].data
                    self.recordLossFile.write(methodName+' '+'loss_bbox %.8f' %
                                              net.blobs['loss_bbox'].data+' '+'loss_cls %.8f' %
                                              net.blobs['loss_cls'].data+' '+'rpn_cls_loss %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+'rpn_loss_bbox %.8f' %
                                              net.blobs['rpn_loss_bbox'].data+' '+'all_loss %.8f\n' %
                                              allLoss)
                elif self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_fuse_multianchor-ms-rpns':
                    if cfg.TRAIN.MULTI_SCALE_RPN_NUM == 3:
                        allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + net.blobs['rpn_cls_loss_conv5_3'].data + \
                                  net.blobs['rpn_loss_bbox_conv5_3'].data + net.blobs['rpn_cls_loss_conv4_3'].data + \
                                  net.blobs['rpn_loss_bbox_conv4_3'].data + net.blobs['rpn_cls_loss_conv3_3'].data + \
                                  net.blobs['rpn_loss_bbox_conv3_3'].data

                        self.recordLossFile.write(methodName+' '+'loss_bbox %.8f' %
                                                  net.blobs['loss_bbox'].data+' '+'loss_cls %.8f' %
                                                  net.blobs['loss_cls'].data+' '+'rpn_cls_loss_conv5_3 %.8f' %
                                                  net.blobs['rpn_cls_loss_conv5_3'].data+' '+'rpn_loss_bbox_conv5_3 %.8f' %
                                                  net.blobs['rpn_loss_bbox_conv5_3'].data+' '+'rpn_cls_loss_conv4_3 %.8f' %
                                                  net.blobs['rpn_cls_loss_conv4_3'].data+' '+'rpn_loss_bbox_conv4_3 %.8f' %
                                                  net.blobs['rpn_loss_bbox_conv4_3'].data+' '+'rpn_cls_loss_conv3_3 %.8f' %
                                                  net.blobs['rpn_cls_loss_conv3_3'].data+' '+'rpn_loss_bbox_conv3_3 %.8f' %
                                                  net.blobs['rpn_loss_bbox_conv3_3'].data+' '+'all_loss %.8f\n' %
                                                  allLoss)
                    else:
                        allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + net.blobs['rpn_cls_loss_conv5_3'].data + \
                                  net.blobs['rpn_loss_bbox_conv5_3'].data + net.blobs['rpn_cls_loss_conv4_3'].data + \
                                  net.blobs['rpn_loss_bbox_conv4_3'].data

                        self.recordLossFile.write(methodName + ' ' + 'loss_bbox %.8f' %
                                                  net.blobs['loss_bbox'].data + ' ' + 'loss_cls %.8f' %
                                                  net.blobs['loss_cls'].data + ' ' + 'rpn_cls_loss_conv5_3 %.8f' %
                                                  net.blobs[
                                                      'rpn_cls_loss_conv5_3'].data + ' ' + 'rpn_loss_bbox_conv5_3 %.8f' %
                                                  net.blobs[
                                                      'rpn_loss_bbox_conv5_3'].data + ' ' + 'rpn_cls_loss_conv4_3 %.8f' %
                                                  net.blobs[
                                                      'rpn_cls_loss_conv4_3'].data + ' ' + 'rpn_loss_bbox_conv4_3 %.8f' %
                                                  net.blobs[
                                                      'rpn_loss_bbox_conv4_3'].data + ' ' + 'all_loss %.8f\n' %
                                                  allLoss)
                else:
                      raise IOError(('The method is not supported by wider_face.\n'))
            elif trainDB == 'face_plus':
                  if self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end' or \
                     self.solver_param.snapshot_prefix == 'ResNet-50_faster_rcnn_end2end_with_multianchor' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_multianchor':
                    allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + \
                              net.blobs['rpn_cls_loss'].data + net.blobs['rpn_loss_bbox'].data + \
                              net.blobs['loss_keyPoint'].data + net.blobs['loss_ethnicity'].data + \
                              net.blobs['loss_gender'].data + net.blobs['loss_age'].data
                    self.recordLossFile.write(methodName+' '+'loss_bbox %.8f' %
                                              net.blobs['loss_bbox'].data+' '+'loss_cls %.8f' %
                                              net.blobs['loss_cls'].data+' '+'rpn_cls_loss %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+'rpn_loss_bbox %.8f' %
                                              net.blobs['rpn_loss_bbox'].data+' '+'loss_ethnicity %.8f' %
                                              net.blobs['loss_ethnicity'].data+' '+'loss_gender %.8f' %
                                              net.blobs['loss_gender'].data+' '+'loss_age %.8f' %
                                              net.blobs['loss_age'].data+' '+'loss_keyPoint %.8f' %
                                              net.blobs['loss_keyPoint'].data+' '+'all_loss %.8f\n' %
                                              allLoss)
                    self.recordLossFile.write('%s\n' % cfg.TRAIN.VISUAL_ANCHORS_IMG)
                  elif self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_multianchor_frozen':
                      allLoss = net.blobs['loss_ethnicity'].data + net.blobs['loss_gender'].data + net.blobs['loss_age'].data
                      self.recordLossFile.write(methodName+' '+'loss_ethnicity %.8f' %
                                                net.blobs['loss_ethnicity'].data+' '+'loss_gender %.8f' %
                                                net.blobs['loss_gender'].data+' '+'loss_age %.8f' %
                                                net.blobs['loss_age'].data+' '+'all_loss %.8f\n' %
                                                allLoss)
                      self.recordLossFile.write('%s\n' % cfg.TRAIN.VISUAL_ANCHORS_IMG)
                  else:
                      raise IOError(('The method is not supported by face_plus.\n'))
            elif trainDB == 'threeHusFace' or trainDB == 'aflw' or trainDB == 'aflw-pifa' or \
                            trainDB == 'cofw':
                  if self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_multianchor' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_fuse_multianchor':
                    allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + \
                              net.blobs['rpn_cls_loss'].data + net.blobs['rpn_loss_bbox'].data + \
                              net.blobs['loss_keyPoint'].data
                    self.recordLossFile.write(methodName+' '+'loss_bbox %.8f' %
                                              net.blobs['loss_bbox'].data+' '+'loss_cls %.8f' %
                                              net.blobs['loss_cls'].data+' '+'rpn_cls_loss %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+'rpn_loss_bbox %.8f' %
                                              net.blobs['rpn_loss_bbox'].data+' '+'loss_keyPoint %.8f' %
                                              net.blobs['loss_keyPoint'].data+' '+'all_loss %.8f\n' %
                                              allLoss)
                    self.recordLossFile.write('%s\n' % cfg.TRAIN.VISUAL_ANCHORS_IMG)
                  elif self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_fuse_multianchor_frozen' or \
                       self.solver_param.snapshot_prefix == 'VGG16_ms-fpn_frozen' or \
                       self.solver_param.snapshot_prefix == 'VGG16_ms-rpn_frozen':
                      allLoss = net.blobs['loss_keyPoint'].data
                      self.recordLossFile.write(methodName+' '+'loss_keyPoint %.8f' %
                                                  net.blobs['loss_keyPoint'].data+' '+'all_loss %.8f\n' %
                                                  allLoss)
                      self.recordLossFile.write('%s\n' % cfg.TRAIN.VISUAL_ANCHORS_IMG)
                  else:
                      raise IOError(('The %s is not supported by face_plus.\n' % self.solver_param.snapshot_prefix))
            elif trainDB == 'morph':
                  if self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end' or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_multianchor'or \
                     self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_fuse_multianchor':
                    allLoss = net.blobs['loss_bbox'].data + net.blobs['loss_cls'].data + \
                              net.blobs['rpn_cls_loss'].data + net.blobs['rpn_loss_bbox'].data + \
                              net.blobs['loss_ethnicity'].data + \
                              net.blobs['loss_gender'].data + net.blobs['loss_age'].data
                    self.recordLossFile.write(methodName+' '+'loss_bbox %.8f' %
                                              net.blobs['loss_bbox'].data+' '+'loss_cls %.8f' %
                                              net.blobs['loss_cls'].data+' '+'rpn_cls_loss %.8f' %
                                              net.blobs['rpn_cls_loss'].data+' '+'rpn_loss_bbox %.8f' %
                                              net.blobs['rpn_loss_bbox'].data+' '+'loss_ethnicity %.8f' %
                                              net.blobs['loss_ethnicity'].data+' '+'loss_gender %.8f' %
                                              net.blobs['loss_gender'].data+' '+'loss_age %.8f' %
                                              net.blobs['loss_age'].data+' '+'all_loss %.8f\n' %
                                              allLoss)
                    self.recordLossFile.write('%s\n' % cfg.TRAIN.VISUAL_ANCHORS_IMG)
                  elif self.solver_param.snapshot_prefix == 'VGG16_faster_rcnn_end2end_with_fuse_multianchor_frozen':
                      allLoss = net.blobs['loss_ethnicity'].data + net.blobs['loss_gender'].data + net.blobs['loss_age'].data
                      self.recordLossFile.write(methodName+' '+'loss_ethnicity %.8f' %
                                                net.blobs['loss_ethnicity'].data+' '+'loss_gender %.8f' %
                                                net.blobs['loss_gender'].data+' '+'loss_age %.8f' %
                                                net.blobs['loss_age'].data+' '+'all_loss %.8f\n' %
                                                allLoss)
                      self.recordLossFile.write('%s\n' % cfg.TRAIN.VISUAL_ANCHORS_IMG)
                  else:
                      raise IOError(('The method is not supported by face_plus.\n'))
            else:
                raise IOError(('The method is not supported.\n'))
        # visual loss
        if self.solver.iter % visualIter == 0 or self.solver.iter == stopIter:
            if self.solver.iter != stopIter:
                self.recordLossFile.flush()
            with open(filename, 'r') as f:
                lines = f.readlines()
            lossStrList = []
            lossTypeNum = (len(lines[0][:-1].split(' '))-1)/2  # remove name and '/n'
            for i in range(lossTypeNum):
                lossName = lines[0].split(' ')[i*2+1]
                lossStrList.append(lossName)
            # run_plot_loss(filename, lossStrList)
            # run_plot_loss_1(filename, lossStrList)

    def recordMetric(self, version, stopIter, startIter = 1, visualIter = 1):
        # self.solver.iter
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = ('Metric_' + self.solver_param.snapshot_prefix + infix +
                    '_{:s}'.format(version) + '.txt')
        filename = os.path.join(self.output_dir, filename)
        ISSTOP = 0
        if self.solver.iter == startIter:
            self.recordMetricFile = open(filename, 'w+')
        elif self.solver.iter == stopIter:
            self.recordMetricFile.close()
            self.recordMetricFile = None
            ISSTOP = 1
        elif self.solver.iter > stopIter:
            if self.recordMetricFile is not None:
                self.recordMetricFile.close()
                self.recordMetricFile = None
                ISSTOP = 1
            else:
                return
        # elif self.solver.iter+startIter > self.max_iters-1:
        #     ISSTOP = 1
        elif self.solver.iter < stopIter and self.recordMetricFile is None:
            raise IOError(('recordLossFile is None before stop.\n'))

        if self.recordMetricFile is not None:
            net = self.solver.net
            # net pre-processing
            scale_bbox_params_faster_rcnn = (cfg.TRAIN.BBOX_REG and
                                 cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                                 net.params.has_key('bbox_pred'))

            scale_bbox_params_rfcn = (cfg.TRAIN.BBOX_REG and
                                 cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                                 net.params.has_key('rfcn_bbox'))

            scale_bbox_params_rpn = (cfg.TRAIN.RPN_NORMALIZE_TARGETS and
                                     net.params.has_key('rpn_bbox_pred'))

            if scale_bbox_params_faster_rcnn:
                # save original values
                orig_0 = net.params['bbox_pred'][0].data.copy()
                orig_1 = net.params['bbox_pred'][1].data.copy()

                # scale and shift with bbox reg unnormalization; then save snapshot
                net.params['bbox_pred'][0].data[...] = \
                        (net.params['bbox_pred'][0].data *
                         self.bbox_stds[:, np.newaxis])
                net.params['bbox_pred'][1].data[...] = \
                        (net.params['bbox_pred'][1].data *
                         self.bbox_stds + self.bbox_means)

            if scale_bbox_params_rpn:
                orig_0 = net.params['rpn_bbox_pred'][0].data.copy()
                orig_1 = net.params['rpn_bbox_pred'][1].data.copy()
                num_anchor = orig_0.shape[0] / 4
                # scale and shift with bbox reg unnormalization; then save snapshot
                self.rpn_means = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_MEANS),
                                          num_anchor)
                self.rpn_stds = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_STDS),
                                         num_anchor)
                net.params['rpn_bbox_pred'][0].data[...] = \
                    (net.params['rpn_bbox_pred'][0].data *
                     self.rpn_stds[:, np.newaxis, np.newaxis, np.newaxis])
                net.params['rpn_bbox_pred'][1].data[...] = \
                    (net.params['rpn_bbox_pred'][1].data *
                     self.rpn_stds + self.rpn_means)

            if scale_bbox_params_rfcn:
                # save original values
                orig_0 = net.params['rfcn_bbox'][0].data.copy()
                orig_1 = net.params['rfcn_bbox'][1].data.copy()
                repeat = orig_1.shape[0] / self.bbox_means.shape[0]

                # scale and shift with bbox reg unnormalization; then save snapshot
                net.params['rfcn_bbox'][0].data[...] = \
                        (net.params['rfcn_bbox'][0].data *
                         np.repeat(self.bbox_stds, repeat).reshape((orig_1.shape[0], 1, 1, 1)))
                net.params['rfcn_bbox'][1].data[...] = \
                        (net.params['rfcn_bbox'][1].data *
                         np.repeat(self.bbox_stds, repeat) + np.repeat(self.bbox_means, repeat))


            tempModelPath = ('Temp_' + self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '_{:s}'.format(version) + '.caffemodel')
            tempModelPath = os.path.join(self.output_dir, tempModelPath)
            prototxt = cfg.TRAIN.METRIC_FILE
            net.save(str(tempModelPath))

            methodType = self.solver_param.snapshot_prefix.split('_')[1]
            # print 'modelPath:%s\nprototxt:%s\nmethodType:%s' % (tempModelPath, prototxt, methodType)
            methodName = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '_{:s}'.format(version))
            # ap = generate_result_wider_val(tempModelPath, prototxt, methodName, methodType)[0]
            # metric_str = generate_result_voc_val(tempModelPath, prototxt, methodName, methodType)
            if cfg.TRAIN.FrozenTraing:
                if not cfg.TRAIN.CacadeFC:
                    metric_str = generate_result_gen_val(tempModelPath, prototxt, methodName,
                                                     cfg.TRAIN.TestDataSet, cfg.TRAIN.ValImgList)
                else:
                    metric_str = generate_result_gen_val(tempModelPath, prototxt, methodName,
                                                     cfg.TRAIN.TestDataSet, cfg.TRAIN.ValImgList, modelType='frozen')
            else:
                if not cfg.TRAIN.CacadeFC:
                    metric_str = generate_result_gen_val(tempModelPath, prototxt, methodName,
                                                     cfg.TRAIN.TestDataSet, cfg.TRAIN.ValImgList)
                else:
                    # Test
                    # tempModelPath = '/data5/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/face_plus/model/VGG16_faster_rcnn_end2end_with_multianchor_v3-7_1_fc_hd1_2_iter_60000.caffemodel'
                    # prototxt = '/data5/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/models/face_plus/VGG16/faster_rcnn_end2end/68kp+age+race+sex/test_multianchor_v3-7_t3.prototxt'
                    # methodName = 'test'
                    # cfg.TRAIN.ValImgList = '/data5/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/Face_plus/miniTest.txt'
                    metric_str = generate_result_gen_val(tempModelPath, prototxt, methodName,
                                                         cfg.TRAIN.TestDataSet, cfg.TRAIN.ValImgList, modelType='cascade')
            os.remove(tempModelPath)
            # self.recordMetricFile.write(methodName+' '+'ap %.8f\n' % ap)
            self.recordMetricFile.write(methodName+' '+'%s\n' % metric_str)
            # net pro-processing
            if scale_bbox_params_faster_rcnn:
                # restore net to original state
                net.params['bbox_pred'][0].data[...] = orig_0
                net.params['bbox_pred'][1].data[...] = orig_1
            if scale_bbox_params_rfcn:
                # restore net to original state
                net.params['rfcn_bbox'][0].data[...] = orig_0
                net.params['rfcn_bbox'][1].data[...] = orig_1
            if scale_bbox_params_rpn:
                # restore net to original state
                net.params['rpn_bbox_pred'][0].data[...] = orig_0
                net.params['rpn_bbox_pred'][1].data[...] = orig_1

        # visual loss
        if self.solver.iter % visualIter == 0 or ISSTOP:
            if not ISSTOP:
                self.recordMetricFile.flush()
            else:
                if self.recordMetricFile is not None:
                    self.recordMetricFile.close()
                    self.recordMetricFile = None
            # with open(filename, 'r') as f:
            #     lines = f.readlines()
            # metricTypeNum = len(lines[0][:-1].split(' '))-1  # remove name and '/n'
            # metricNames = [line.split(':')[0] for line in lines[0][:-1].split(' ')[1:]]
            # metricList = {}
            # plt.figure(1)
            # # plt.title('%s Metric' % self.solver_param.snapshot_prefix)
            # plotStartIter = startIter - cfg.TRAIN.METRIC_ITERS
            # metricNum = len(cfg.TRAIN.TestMetrics)
            # plotRawNum = int(math.ceil(metricNum / 3.0))
            # plotColNum = metricNum if metricNum < 3 else 3
            # plotIndex = 1
            # for i in range(metricTypeNum):
            #     metricName = metricNames[i]
            #     if metricName in cfg.TRAIN.TestMetrics:
            #         ax = plt.subplot(plotRawNum, plotColNum, plotIndex)
            #         metricList[metricName] = [line[:-1].split(' ')[1:][i].split(':')[1] for line in lines]
            #         plt.plot(plotStartIter + cfg.TRAIN.METRIC_ITERS*np.arange(1, len(lines)+1), metricList[metricName], label=metricName)
            #         plt.xlabel('iteration')
            #         plt.ylabel(metricName)
            #         plt.legend()
            #         plt.tight_layout()
            #         plotIndex = plotIndex + 1
            # # metricName = metricNames[-1]
            # # metricList[metricName] = [line[:-1].split(' ')[1:][-1].split(':')[1] for line in lines]
            # # plotStartIter = startIter - cfg.TRAIN.METRIC_ITERS
            # # plt.plot(plotStartIter + cfg.TRAIN.METRIC_ITERS*np.arange(1, len(lines)+1), metricList[metricName], label=metricName)
            #
            # imagefile = ('VMetric_' + self.solver_param.snapshot_prefix + infix +
            #              '_{:s}'.format(version) + '.jpg')
            # imagefile = os.path.join(self.output_dir, imagefile)
            # # plt.show()
            # plt.savefig(imagefile, dpi=300)
            # plt.close('all')
            # exit(1)

    def snapshot(self, version):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params_faster_rcnn = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        scale_bbox_params_rfcn = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('rfcn_bbox'))

        scale_bbox_params_rpn = (cfg.TRAIN.RPN_NORMALIZE_TARGETS and
                                 net.params.has_key('rpn_bbox_pred'))

        if scale_bbox_params_faster_rcnn:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if scale_bbox_params_rpn:
            orig_0 = net.params['rpn_bbox_pred'][0].data.copy()
            orig_1 = net.params['rpn_bbox_pred'][1].data.copy()
            num_anchor = orig_0.shape[0] / 4
            # scale and shift with bbox reg unnormalization; then save snapshot
            self.rpn_means = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_MEANS),
                                      num_anchor)
            self.rpn_stds = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_STDS),
                                     num_anchor)
            net.params['rpn_bbox_pred'][0].data[...] = \
                (net.params['rpn_bbox_pred'][0].data *
                 self.rpn_stds[:, np.newaxis, np.newaxis, np.newaxis])
            net.params['rpn_bbox_pred'][1].data[...] = \
                (net.params['rpn_bbox_pred'][1].data *
                 self.rpn_stds + self.rpn_means)

        if scale_bbox_params_rfcn:
            # save original values
            orig_0 = net.params['rfcn_bbox'][0].data.copy()
            orig_1 = net.params['rfcn_bbox'][1].data.copy()
            repeat = orig_1.shape[0] / self.bbox_means.shape[0]

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['rfcn_bbox'][0].data[...] = \
                    (net.params['rfcn_bbox'][0].data *
                     np.repeat(self.bbox_stds, repeat).reshape((orig_1.shape[0], 1, 1, 1)))
            net.params['rfcn_bbox'][1].data[...] = \
                    (net.params['rfcn_bbox'][1].data *
                     np.repeat(self.bbox_stds, repeat) + np.repeat(self.bbox_means, repeat))

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        baseFileName = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '_{:s}'.format(version))  # + '.caffemodel'
        filename = os.path.join(self.output_dir, baseFileName + '.caffemodel')

        if cfg.TRAIN.WITH_SOLVERSTATE:
            self.solver.snapshot()
        else:
            net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params_faster_rcnn:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        if scale_bbox_params_rfcn:
            # restore net to original state
            net.params['rfcn_bbox'][0].data[...] = orig_0
            net.params['rfcn_bbox'][1].data[...] = orig_1
        if scale_bbox_params_rpn:
            # restore net to original state
            net.params['rpn_bbox_pred'][0].data[...] = orig_0
            net.params['rpn_bbox_pred'][1].data[...] = orig_1

        return filename

    def train_model(self, max_iters, version, trainDB):
        """Network training loop."""
        self.max_iters = max_iters
        timer = Timer()
        model_paths = []

        # save when iter = 0
        # self.last_snapshot_iter = self.solver.iter
        # model_paths.append(self.snapshot(version))
        # exit(1)

        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            # record loss and metric every particular iter
            self.recordLoss(version, max_iters-1, trainDB, startIter=self.start_loss_iter, visualIter=100)  # max_iters-1 100 4 2
            # self.start_metric_iter = 0  # debug
            if self.solver.iter % cfg.TRAIN.METRIC_ITERS == 0:
                self.recordMetric(version, max_iters-1, startIter=self.start_metric_iter, visualIter=1)  # max_iters-1 1 5
                # exit(1)
            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self.last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot(version))

            # parameterPath = '/data6/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/wider_face/temp1.txt'
            # with open(parameterPath, 'w') as f:
            #     for paramName in self.solver.net.params.keys():
            #         paramNum = len(self.solver.net.params[paramName])
            #         f.write('%s' % paramName)
            #         for i in range(paramNum):
            #             paramName_v = str(self.solver.net.params[paramName][i].data.ravel()[0:8])
            #             f.write(' %s' % paramName_v)
            #         f.write('\n')
            # exit(1)

        if self.last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    print 'orgirnal num of image: %d' % len(imdb.roidb)
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()  # including gt_roidb and flipped_roidb
        print 'the num of image after flipped: %d' % len(imdb.roidb)

    if cfg.TRAIN.MULTI_SCALE:
        print 'Appending training examples with multi-scale %s...' % str(cfg.TRAIN.MULTI_SCALE_LIST)
        imdb.append_multiScaled_images()  # including gt_roidb and flipped_roidb
        print 'the num of image after multiscale: %d' % len(imdb.roidb)

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)  # extend more attributes for roidb
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb, output_dir, version, trainDB,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(solver_prototxt, roidb, output_dir, version,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters, version, trainDB)
    print 'done solving'
    return model_paths
