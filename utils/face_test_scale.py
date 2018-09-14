#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import _init_ms_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, im_detect_facePlus, im_detect_facePlus_v1
# from rpn.generate import im_proposals
import rpn.generate  # for call im_proposals in face_test.py
from fast_rcnn.nms_wrapper import nms
import math
import numpy as np
import caffe, os, sys, cv2
import linecache
from datasets.voc_eval import voc_ap, parse_rec
from utils.timer import Timer
import matplotlib.pyplot as plt
import shutil
from ProgressBar import *
import scipy.io as sio
# from utils.imdb_explore.FaceImage import FaceImage
from utils.MultiAttributeDB.FaceImage import FaceImage
from mylab.draw import *
from utils.threeHus_face.plot_results_python import plot_results, plot_results_gen

CLASSES = ('__background__',
           'face')
threeHusOtherDBDir = cfg.ROOT_DIR + '/data/DB/face/300-w_face/otherDB/'
threeHusTestDir = cfg.ROOT_DIR + '/data/DB/face/300-w_face/300w_cropped/'
threeHusOtherMetricsDir = cfg.ROOT_DIR + '/data/DB/face/300-w_face/300W_results/'
facePlusValImgList = cfg.ROOT_DIR + '/data/DB/face/Face_plus/test.txt'
matFile = cfg.ROOT_DIR + '/data/DB/face/wider/wider_face_val-v7.mat'
widerValDir = cfg.ROOT_DIR + '/data/DB/face/wider/val/'
testImgList = cfg.ROOT_DIR + '/data/DB/face/FDDB/FDDB-fold-all.txt'
# Annotations_new_v2
widerValImgList = cfg.ROOT_DIR + '/data/DB/object/voc_wider/ImageSets/Main/val.txt'
widerValAnnoPath = cfg.ROOT_DIR + '/data/DB/object/voc_wider/Annotations/{:s}.xml'
filterImgList = cfg.ROOT_DIR + '/data/DB/face/FDDB/FDDB-filter-all.txt'
detectFilePath = cfg.ROOT_DIR + '/data/DB/face/FDDB/temp/temp_2_result.txt'
vocDetectFilePath = cfg.ROOT_DIR + '/data/DB/face/FDDB/temp/temp_voc_2_result.txt'
fuseDetectFilePath = ''
testImgDir = cfg.ROOT_DIR + '/data/DB/face/FDDB/'
widerValImgDir = cfg.ROOT_DIR + "/data/DB/object/voc_wider/JPEGImages/"
annoFilePath = cfg.ROOT_DIR + '/data/DB/face/FDDB/FDDB-ellipseList-all.txt'
# Annotations_new_v2
imageSaveDir = ''
imfpFilePath = ''
sortedImfpFilePath = ''
sortedImtpFilePath = ''

class customError(StandardError):
    pass


def generate_result_fddb(modelPath, prototxt, modelType='NORMAL', recompute=1,
                         CONF_THRESH=0.8, NMS_THRESH=0.3, visual=0, recordIMFP=0,
                         Adapter=0):
    # load initial parameter
    gpu_id = 0
    cfg.TEST.HAS_RPN = True
    # initialize net
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    # detect and record result
    im_names = linecache.getlines(testImgList)
    oriImgInfos = loadResultFile(annoFilePath)
    if not os.path.exists(detectFilePath) or recompute:
        resultlist = open(detectFilePath, 'w')
        for im_name in im_names:
            # Load the demo image
            image_name = im_name[:-1]
            im_path = os.path.join(testImgDir, image_name + '.jpg')
            im = cv2.imread(im_path)
            # Detect all object classes and regress object bounds
            if modelType == 'rpn':
                boxes, scores = rpn.generate.im_proposals(net, im)
            else:
                scores, boxes = im_detect(net, im)

            for cls_ind, cls in enumerate(CLASSES[1:]):
                if modelType != 'rpn':
                    cls_ind += 1  # because we skipped background
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                # record result
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                # adapt num of detection to gt
                if Adapter:
                    IM_GT_NUM = len(oriImgInfos[image_name])
                    inds = np.argsort(-dets[inds, -1])
                    inds = inds[:IM_GT_NUM]

                resultlist.write(image_name + '\n')
                resultlist.write(str(len(inds)) + '\n')
                if len(inds) == 0:
                    continue
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    resultlist.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                     str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
        resultlist.close()

    # load detect file
    detImgInfos = loadResultFile(detectFilePath)
    # visual and save record
    if visual:
        visual_result(oriImgInfos, detImgInfos)
    # compute metric
    metric = cal_ap_mp(oriImgInfos, detImgInfos, im_names, recordIMFP=recordIMFP)
    return metric


def generate_result_wider_val(modelPath, prototxt, methodName, modelType='NORMAL', recompute=1,
                              CONF_THRESH=0.8, NMS_THRESH=0.3, visual=0, recordIMFP=0,
                              Adapter=0):
    widerValDetectFilePath = cfg.ROOT_DIR + '/data/DB/face/temp/temp_%s.txt' % methodName
    # load initial parameter
    gpu_id = 1
    cfg.TEST.HAS_RPN = True
    # initialize net
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    # detect and record result
    widerValImNames = linecache.getlines(widerValImgList)
    widerValAnnoImgInfos = loadResultFile_voc(widerValAnnoPath, widerValImNames)
    # detect and record in voc format
    if not os.path.exists(widerValDetectFilePath) or recompute:
        det_model_voc(prototxt, modelPath, modelType, widerValImNames, widerValDetectFilePath,
                      widerValImgDir, CONF_THRESH, NMS_THRESH)

    # load detect file
    widerValDetImgInfos = loadDetectFile_voc(widerValDetectFilePath, widerValImNames)
    # visual and save record
    if visual:
        visual_result_voc(widerValAnnoImgInfos, widerValDetImgInfos, detImageSaveDir, widerValImgDir)
    # compute metric
    metric = cal_ap_mp_voc(widerValAnnoImgInfos, widerValDetectFilePath, widerValImNames, recordIMFP=recordIMFP)
    return metric


def generate_result_voc_val(modelPath, prototxt, methodName, modelType='NORMAL', recompute=1,
                            CONF_THRESH=0.8, NMS_THRESH=0.3, visual=0, recordIMFP=0,
                            Adapter=0):
    vocValDetectFilePath = cfg.ROOT_DIR + '/data/DB/face/temp/temp_%s.txt' % methodName
    # load initial parameter
    gpu_id = 2
    cfg.TEST.HAS_RPN = True
    # initialize net
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    # detect and record result
    vocValImNames = linecache.getlines(cfg.TRAIN.ValImgList)
    vocValAnnoImgInfos = loadResultFile_voc(cfg.TRAIN.ValAnnoPath, vocValImNames)
    # detect and record in voc format
    if not os.path.exists(vocValDetectFilePath) or recompute:
        det_model_voc(prototxt, modelPath, modelType, vocValImNames, vocValDetectFilePath,
                      cfg.TRAIN.vocValImgDir, CONF_THRESH, NMS_THRESH)

    # load detect file
    vocValDetImgInfos = loadDetectFile_voc(vocValDetectFilePath, vocValImNames)
    # visual and save record
    if visual:
        visual_result_voc(vocValAnnoImgInfos, vocValDetImgInfos, detImageSaveDir, cfg.TRAIN.vocValImgDir)
    # if
    # compute metric
    metric = cal_ap_mp_voc(vocValAnnoImgInfos, vocValDetectFilePath, vocValImNames, recordIMFP=recordIMFP)
    return metric[-1]


def generate_result_gen_val(modelPath, prototxt, methodName, testDataSet, valImgList,
                            modelType='NORMAL', recompute=1, CONF_THRESH=0.8, NMS_THRESH=0.3, visual=0, recordIMFP=0):
    vocValDetectFilePath = cfg.ROOT_DIR + '/data/DB/face/temp/temp_%s.txt' % methodName
    # load initial parameter
    gpu_id = 0
    cfg.TEST.HAS_RPN = True
    # initialize net
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    # net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    if testDataSet == 'wider':
        # detect and record result
        vocValImNames = linecache.getlines(cfg.TRAIN.ValImgList)
        vocValAnnoImgInfos = loadResultFile_voc(cfg.TRAIN.ValAnnoPath, vocValImNames)
        # detect and record in voc format
        if not os.path.exists(vocValDetectFilePath) or recompute:
            # det_model_voc(prototxt, modelPath, modelType, vocValImNames, vocValDetectFilePath,
            #               cfg.TRAIN.vocValImgDir, CONF_THRESH, NMS_THRESH)
            det_model_voc_pyramid(prototxt, modelPath, modelType, vocValImNames, vocValDetectFilePath,
                                  cfg.TRAIN.vocValImgDir, CONF_THRESH, NMS_THRESH, includeRPN=cfg.TRAIN.METRIC_RPN)
        if cfg.TRAIN.METRIC_RPN:
            vocValDetImgInfos = loadDetectFile_voc(vocValDetectFilePath, vocValImNames)
            vocValDetImgInfosRPN = loadDetectFile_voc(vocValDetectFilePath.replace('.txt', '_RPN.txt'), vocValImNames)
            # compute metric
            metric = cal_ap_mp_voc(vocValAnnoImgInfos, vocValDetectFilePath, vocValImNames, recordIMFP=recordIMFP)
            metricRPN = cal_ap_mp_voc(vocValAnnoImgInfos, vocValDetectFilePath.replace('.txt', '_RPN.txt'), vocValImNames, recordIMFP=recordIMFP)
            metricRPN_str = ' '.join(['RPN_%s' % i for i in metricRPN[-1].split(' ')[1:]])
            metric[-1] = metric[-1] + ' ' + metricRPN_str
        else:
            # load detect file
            vocValDetImgInfos = loadDetectFile_voc(vocValDetectFilePath, vocValImNames)
            # visual and save record
            if visual:
                visual_result_voc(vocValAnnoImgInfos, vocValDetImgInfos, detImageSavePath, cfg.TRAIN.vocValImgDir)
            # if
            # compute metric
            metric = cal_ap_mp_voc(vocValAnnoImgInfos, vocValDetectFilePath, vocValImNames, recordIMFP=recordIMFP)
    elif testDataSet == 'Face_Plus':
        # detect and record result
        facePlusValImPaths = linecache.getlines(valImgList)
        facePlusValAnnoImgInfos = [FaceImage(facePlusValImPath[:-1]) for facePlusValImPath in facePlusValImPaths]

        # detect and record in voc format
        if not os.path.exists(vocValDetectFilePath) or recompute:
            det_model_facePlus_pyramid(prototxt, modelPath, modelType, facePlusValImPaths, vocValDetectFilePath,
                                       CONF_THRESH, NMS_THRESH)

        # load detect file
        print 'load detect file'
        facePlusValDetImgInfos = loadDetectFile_facePlus(vocValDetectFilePath, facePlusValImPaths)

        # visual and save record
        if visual:
            visual_result_facePlus(facePlusValAnnoImgInfos, facePlusValDetImgInfos, detImageSaveDir)
        # compute metric
        metric = cal_ap_mp_facePlus(facePlusValAnnoImgInfos, vocValDetectFilePath, recordIMFP=recordIMFP)

    elif testDataSet == 'Face_Plus_v1':
        # detect and record result
        facePlusValImPaths = linecache.getlines(valImgList)
        facePlusValAnnoImgInfos = [FaceImage(facePlusValImPath[:-1]) for facePlusValImPath in facePlusValImPaths]

        # detect and record in voc format
        if not os.path.exists(vocValDetectFilePath) or recompute:
            det_model_facePlus_v1_pyramid(prototxt, modelPath, modelType, facePlusValImPaths, vocValDetectFilePath,
                                       CONF_THRESH, NMS_THRESH)

        # load detect file
        print 'load detect file'
        facePlusValDetImgInfos = loadDetectFile_facePlus_v1(vocValDetectFilePath, facePlusValImPaths)

        # visual and save record
        if visual:
            visual_result_facePlus_v1(facePlusValAnnoImgInfos, facePlusValDetImgInfos, detImageSaveDir)
        # compute metric
        metric = cal_ap_mp_facePlus_v1(facePlusValAnnoImgInfos, vocValDetectFilePath, recordIMFP=recordIMFP)


    elif testDataSet == 'threeHusFace':
        threeHusTestDir = valImgList
        threeHusValDetectFilePath = vocValDetectFilePath
        threeHusValAnnoImgInfos = loadResultFile_300w_face(threeHusTestDir)
        threeHusValImPaths = threeHusValAnnoImgInfos.keys()
        # detect and record in facePlus format
        if not os.path.exists(threeHusValDetectFilePath) or recompute:
            # det_model_voc
            det_model_facePlus_pyramid(prototxt, modelPath, modelType, threeHusValImPaths, threeHusValDetectFilePath,
                                       CONF_THRESH, NMS_THRESH)
        # exit(0)
        # load detect file
        print 'load detect file'
        threeHusValDetImgInfos = loadDetectFile_facePlus(threeHusValDetectFilePath, threeHusValImPaths)
        # visual and save record
        if visual:
            visual_result_300w_face(threeHusValAnnoImgInfos, threeHusValDetImgInfos, detImageSaveDir)
        # exit(0)
        # compute metric
        metric = cal_cer_threeHus(threeHusValAnnoImgInfos, threeHusValDetImgInfos)

    else:
        print 'testDataset is invaiid'
        exit()
    return metric[-1]


def generate_result_v1(modelPath, prototxt, testDataSet, modelType='NORMAL', recompute=1,
                       CONF_THRESH=0.8, NMS_THRESH=0.3, visual=0, recordIMFP=0,
                       Adapter=0, segModelPath=None, segPrototxt=None, segWeight=0.1,
                       strategyType=None, scales=[0], metricFilePath=None, tpfpFilePath=None,
                       includeRPN=0):
    # load initial parameter
    gpu_id = 0
    cfg.TEST.HAS_RPN = True
    cfg.GPU_ID = gpu_id
    # initialize net
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    if testDataSet == 'wider':
        # detect and record result
        widerValImNames = linecache.getlines(widerValImgList)
        widerValAnnoImgInfos = loadResultFile_voc(widerValAnnoPath, widerValImNames)

        # detect and record in wider face format
        if not os.path.exists(widerDetectSaveDir) or recompute:
            # det_model_wider
            det_model_wider_pyramid(prototxt, modelPath, modelType, matFile, widerDetectSaveDir,
                                    CONF_THRESH, NMS_THRESH, scales=scales)
        # detect and record in voc format
        if not os.path.exists(widerValDetectFilePath) or recompute:
            # det_model_voc
            det_model_voc_pyramid(prototxt, modelPath, modelType, widerValImNames, widerValDetectFilePath,
                                  widerValImgDir, CONF_THRESH, NMS_THRESH, scales=scales, includeRPN=includeRPN)
        # exit(1)
        if includeRPN:
            # load detect file
            print 'load detect file'
            widerValDetImgInfos = loadDetectFile_voc(widerValDetectFilePath, widerValImNames)
            widerValDetImgInfosRPN = loadDetectFile_voc(widerValDetectFilePath.replace('.txt', '_RPN.txt'), widerValImNames)
            # visual and save record
            if visual:
                visual_result_voc(widerValAnnoImgInfos, widerValDetImgInfos, detImageSaveDir, widerValImgDir)
                visual_result_voc(widerValAnnoImgInfos, widerValDetImgInfosRPN, detImageSaveDir + '_RPN', widerValImgDir)
            # compute metric
            metric = cal_ap_mp_voc(widerValAnnoImgInfos, widerValDetectFilePath, widerValImNames, recordIMFP=0)
            metricRPN = cal_ap_mp_voc(widerValAnnoImgInfos, widerValDetectFilePath.replace('.txt', '_RPN.txt'), widerValImNames, recordIMFP=recordIMFP)
            # record metric
            record_result(metric, metricFilePath, tpfpFilePath)
            record_result(metricRPN, metricFilePath, tpfpFilePath.replace('.jpg', '_RPN.jpg'), includeRPN=True)
            # exit(1)
        else:
            # load detect file
            print 'load detect file'
            widerValDetImgInfos = loadDetectFile_voc(widerValDetectFilePath, widerValImNames)
            # visual and save record
            if visual:
                visual_result_voc(widerValAnnoImgInfos, widerValDetImgInfos, detImageSaveDir, widerValImgDir)
            # compute metric
            metric = cal_ap_mp_voc(widerValAnnoImgInfos, widerValDetectFilePath, widerValImNames, recordIMFP=recordIMFP)
            # record metric
            record_result(metric, metricFilePath, tpfpFilePath)

    elif testDataSet == 'FDDB':
        # detect fddb face
        im_names = linecache.getlines(testImgList)
        oriImgInfos = loadResultFile(annoFilePath)
        if not os.path.exists(detectFilePath) or recompute:
            print 'detect face'
            # det_model(prototxt, modelPath, modelType, im_names, detectFilePath, Adapter, oriImgInfos,
            #           CONF_THRESH, NMS_THRESH)
            det_model_voc_pyramid(prototxt, modelPath, modelType, im_names, detectFilePath,
                                  testImgDir, CONF_THRESH, NMS_THRESH, scales=scales,
                                  testingDB=testDataSet)

        # load detect file
        print 'load detect file'
        detImgInfos = loadResultFile(detectFilePath)

        if segModelPath is not None:
            # segment face
            print 'segment face'
            segmentInfos = seg_model(segPrototxt, segModelPath, im_names)
            # debug
            # visual_det_seg(detImgInfos, segmentInfos, im_names)
            # fuse result with segmentation
            print 'fuse result with segmentation of segWeight:%03d' % (segWeight * 100)
            detImgInfos = fuseSegResult(detImgInfos, segmentInfos, im_names, segWeight, strategyType)
            # save new detImgInfos
            detImgInfos = saveDetImgInfos(detImgInfos, im_names)
        # visual and save record
        if visual:
            # visual_result_voc(widerValAnnoImgInfos, widerValDetImgInfos, detImageSaveDir, widerValImgDir)
            visual_result(oriImgInfos, detImgInfos, imageSaveDir)
        # compute metric
        metric = cal_ap_mp(oriImgInfos, detImgInfos, im_names, recordIMFP=recordIMFP)
        # record metric
        record_result(metric, metricFilePath, tpfpFilePath)

    elif testDataSet == 'Face_Plus':
        # detect and record result
        facePlusValImPaths = linecache.getlines(facePlusValImgList)
        facePlusValImPaths = [facePlusValImPath[:-1] for facePlusValImPath in facePlusValImPaths]
        facePlusValAnnoImgInfos = [FaceImage(facePlusValImPath) for facePlusValImPath in facePlusValImPaths]
        # detect and record in facePlus format
        if not os.path.exists(facePlusValDetectFilePath) or recompute:
            # det_model_voc
            det_model_facePlus_pyramid(prototxt, modelPath, modelType, facePlusValImPaths, facePlusValDetectFilePath,
                                       CONF_THRESH, NMS_THRESH, scales=scales)
        # load detect file
        print 'load detect file'
        facePlusValDetImgInfos = loadDetectFile_facePlus(facePlusValDetectFilePath, facePlusValImPaths)
        # visual and save record
        if visual:
            visual_result_facePlus(facePlusValAnnoImgInfos, facePlusValDetImgInfos, detImageSaveDir)
        # compute metric
        metric = cal_ap_mp_facePlus(facePlusValAnnoImgInfos, facePlusValDetectFilePath, recordIMFP=recordIMFP)
        # record metric
        record_result_facePlus(metric, metricFilePath, tpfpFilePath)
        # compare metric
        saveFilePath = metricFilePath.replace('.txt', '.jpg')
        MetricsDir = '/'.join(metricFilePath.split('/')[:-1])
        plot_results_gen(MetricsDir, saveFilePath)
    elif testDataSet == 'Face_Plus_v1':
        # detect and record result
        facePlusValImPaths = linecache.getlines(facePlusValImgList)
        facePlusValImPaths = [facePlusValImPath[:-1] for facePlusValImPath in facePlusValImPaths]
        facePlusValAnnoImgInfos = [FaceImage(facePlusValImPath) for facePlusValImPath in facePlusValImPaths]
        # detect and record in facePlus format
        if not os.path.exists(facePlusValDetectFilePath) or recompute:
            # det_model_voc
            det_model_facePlus_v1_pyramid(prototxt, modelPath, modelType, facePlusValImPaths, facePlusValDetectFilePath,
                                       CONF_THRESH, NMS_THRESH, scales=scales)
        # load detect file
        print 'load detect file'
        facePlusValDetImgInfos = loadDetectFile_facePlus_v1(facePlusValDetectFilePath, facePlusValImPaths)
        # visual and save record
        if visual:
            visual_result_facePlus_v1(facePlusValAnnoImgInfos, facePlusValDetImgInfos, detImageSaveDir)
        # compute metric
        metric = cal_ap_mp_facePlus_v1(facePlusValAnnoImgInfos, facePlusValDetectFilePath, recordIMFP=recordIMFP)
        # record metric
        record_result_facePlus_v1(metric, metricFilePath, tpfpFilePath)
        # compare metric
        saveFilePath = metricFilePath.replace('.txt', '.jpg')
        MetricsDir = '/'.join(metricFilePath.split('/')[:-1])
        plot_results_gen(MetricsDir, saveFilePath)
    elif testDataSet == 'threeHusFace':
        threeHusValAnnoImgInfos = loadResultFile_300w_face(threeHusTestDir)
        threeHusValImPaths = threeHusValAnnoImgInfos.keys()
        # detect and record in facePlus format
        if not os.path.exists(threeHusValDetectFilePath) or recompute:
            # det_model_voc
            det_model_facePlus_pyramid(prototxt, modelPath, modelType, threeHusValImPaths, threeHusValDetectFilePath,
                                       CONF_THRESH, NMS_THRESH, scales=scales)
        # exit(0)
        # load detect file
        print 'load detect file'
        threeHusValDetImgInfos = loadDetectFile_facePlus(threeHusValDetectFilePath, threeHusValImPaths)
        # visual and save record
        if visual:
            visual_result_300w_face(threeHusValAnnoImgInfos, threeHusValDetImgInfos, detImageSaveDir)
        # exit(0)
        # compute metric
        metric = cal_cer_threeHus(threeHusValAnnoImgInfos, threeHusValDetImgInfos)
        # record metric
        record_result_threeHus(metric, metricFilePath)
        # compare metric
        threeHusMetricFilePath = os.path.join(threeHusOtherMetricsDir, '300W_v2', metricFilePath.split('/')[-1])
        # shutil.copyfile(metricFilePath, threeHusMetricFilePath)
        saveFilePath = metricFilePath.replace('.txt', '.jpg')
        plot_results(2, threeHusOtherMetricsDir, saveFilePath)
    else:
        print 'testDataset is invaiid'
    return metric


def generate_result_voc(modelPath, prototxt, vocValAnnoPath, vocValImgDir, vocValImgList, vocValDetectFilePath,
                        modelType='NORMAL', recompute=1, CONF_THRESH=0.8, NMS_THRESH=0.3, visual=0, recordIMFP=0):
    # load initial parameter
    gpu_id = 1
    cfg.TEST.HAS_RPN = True
    cfg.GPU_ID = gpu_id
    # initialize net
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    # detect and record result
    vocValImNames = linecache.getlines(vocValImgList)
    vocValAnnoImgInfos = loadResultFile_voc(vocValAnnoPath, vocValImNames)

    # detect and record in voc format
    if not os.path.exists(vocValDetectFilePath) or recompute:
        det_model_voc(prototxt, modelPath, modelType, vocValImNames, vocValDetectFilePath,
                      vocValImgDir, CONF_THRESH, NMS_THRESH)

    # load detect file
    print 'load detect file'
    vocValDetImgInfos = loadDetectFile_voc(vocValDetectFilePath, vocValImNames)

    # visual and save record
    if visual:
        visual_result_voc(vocValAnnoImgInfos, vocValDetImgInfos, detImageSaveDir, vocValImgDir)

    # compute metric
    metric_voc = cal_ap_mp_voc(vocValAnnoImgInfos, vocValDetectFilePath, vocValImNames, recordIMFP=recordIMFP)
    return metric_voc


def saveDetImgInfos(detImgInfos, im_names):
    resultList = open(fuseDetectFilePath, 'w')
    for im_name in im_names:
        image_name = im_name[:-1]
        detImgInfo = detImgInfos[image_name]
        # resultList.write('_'.join(image_name.split('/'))+'\n')
        resultList.write(image_name + '\n')
        resultList.write(str(len(detImgInfo)) + '\n')
        for i, det in enumerate(detImgInfo):
            if adjustBox:
                dets = np.array(det.split(' '), dtype=np.float32)
                dets[3] = dets[3] + dets[3] * adjustRate
                dets[1] = dets[1] - dets[3] * adjustRate
                if dets[1] < 0:
                    dets[3] = dets[3] + dets[1]
                    dets[1] = 0
                det = ' '.join(dets.astype(np.str))
                detImgInfo[i] = det
            resultList.write(det + '\n')
    resultList.close()
    return detImgInfos


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


def det_model(prototxt, modelPath, modelType, im_names, detectFilePath,
              Adapter=0, oriImgInfos=None, CONF_THRESH=0.8, NMS_THRESH=0.3):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    resultList = open(detectFilePath, 'w')
    for im_name in im_names:
        # Load the demo image
        image_name = im_name[:-1]

        # filter specific img
        # Adapter = filter_adapter(image_name)
        # if Adapter:
        #     CONF_THRESH = 0.5
        #     NMS_THRESH = 0.3
        # else:
        #     CONF_THRESH = 0.8
        #     NMS_THRESH = 0.3

        im_path = os.path.join(testImgDir, image_name + '.jpg')
        im = cv2.imread(im_path)
        # Detect all object classes and regress object bounds
        if modelType == 'rpn':
            boxes, scores = rpn.generate.im_proposals(net, im)
        else:
            scores, boxes = im_detect(net, im)

        for cls_ind, cls in enumerate(CLASSES[1:]):
            if modelType != 'rpn':
                cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            # record result
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

            # adapt num of detection to gt
            # if Adapter:
            #     IM_GT_NUM = len(oriImgInfos[image_name])
            #     inds = np.argsort(-dets[inds, -1])
            #     inds = inds[:IM_GT_NUM]

            resultList.write(image_name + '\n')
            resultList.write(str(len(inds)) + '\n')
            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                resultList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                 str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
    resultList.close()


def det_model_voc(prototxt, modelPath, modelType, im_names, vocDetectFilePath,
                  vocValImgDir, CONF_THRESH=0.8, NMS_THRESH=0.3):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    resultList = open(vocDetectFilePath, 'w')
    for im_name in im_names:
        # Load the demo image
        image_name = im_name[:-1]

        im_path = os.path.join(vocValImgDir, image_name + '.jpg')
        im = cv2.imread(im_path)
        # Detect all object classes and regress object bounds
        if modelType == 'rpn':
            boxes, scores = rpn.generate.im_proposals(net, im)
        else:
            scores, boxes = im_detect(net, im)

        for cls_ind, cls in enumerate(CLASSES[1:]):
            if modelType != 'rpn':
                cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            # record result
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                # resultList.write(image_name+'\n')
                # resultList.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2]-bbox[0])+' '+
                #                  str(bbox[3]-bbox[1])+' '+str(score)+'\n')
                resultList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                 format(image_name, score,
                                        bbox[0] + 1, bbox[1] + 1,
                                        bbox[2] + 1, bbox[3] + 1))
    resultList.close()


def seg_model(prototxt, modelPath, im_names):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    segmentInfos = {}
    pbar = ProgressBar(len(im_names))
    for im_name in im_names:
        pbar += 1
        # Load the demo image
        image_name = im_name[:-1]
        im_path = os.path.join(testImgDir, image_name + '.jpg')
        im = cv2.imread(im_path).astype(np.float32)
        im -= np.array((104.00698793, 116.66876762, 122.67891434))
        im = im.transpose((2, 0, 1))  # stick  channel
        # net = caffe.Net('voc-fcn32s/deploy.prototxt', 'voc-fcn32s/fcn32s-heavy-pascal.caffemodel', caffe.TEST)
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *im.shape)
        net.blobs['data'].data[...] = im
        # run net and take argmax for prediction
        # timer = Timer()
        # timer.tic()
        net.forward()
        # timer.toc()
        # print ('total took {:.3f}s').format(timer.total_time)
        bg = net.blobs['score'].data[0][0]
        fg = net.blobs['score'].data[0][1]
        bg = MaxMinNormalization(bg)
        fg = MaxMinNormalization(fg)
        fmb = fg - bg
        # fmb = MaxMinNormalization(fmb)
        out = net.blobs['score'].data[0].argmax(axis=0)
        # return out, bg, fg, fmb
        if strategyType == 's1' or strategyType == 's3' or strategyType == 's4':
            segmentInfos[image_name] = fmb
        elif strategyType == 's2':
            segmentInfos[image_name] = out
    return segmentInfos


def fuseSegResult(detImgInfos, segmentInfos, im_names, segWeight, strategyType):
    for im_name in im_names:
        image_name = im_name[:-1]
        segImgInfo = np.array(segmentInfos[image_name])
        detImgInfo = detImgInfos[image_name]
        detImgBoxs = np.array([detImgBox.split(' ') for detImgBox in detImgInfo], dtype=np.float32)
        for i in range(len(detImgBoxs)):
            x1 = detImgBoxs[i][0].astype(np.int)
            y1 = detImgBoxs[i][1].astype(np.int)
            x2 = (detImgBoxs[i][2] + detImgBoxs[i][0]).astype(np.int)
            y2 = (detImgBoxs[i][3] + detImgBoxs[i][1]).astype(np.int)
            detScore = detImgBoxs[i][4] * (1 - segWeight)
            if strategyType == 's1':
                # strategy 1 for fmg
                segScoreMean = segImgInfo[y1:y2, x1:x2].mean()
                segScore = segScoreMean * segWeight
                newDetScore = detScore + segScore
                detImgBox = np.append(detImgBoxs[i], [segScoreMean, newDetScore])
                detImgInfo[i] = ' '.join(detImgBox.astype(np.str))
            elif strategyType == 's2':
                # strategy 2 for out
                overset = segImgInfo[y1:y2, x1:x2]
                oversetRate = np.sum(overset).astype(np.float32) / (overset.shape[0] * overset.shape[1])
                if oversetRate > overSegThresh:
                    newDetScore = detScore + oversetRate * segWeight
                else:
                    newDetScore = oversetRate
                detImgBox = np.append(detImgBoxs[i], [oversetRate, newDetScore])
                detImgInfo[i] = ' '.join(detImgBox.astype(np.str))
            elif strategyType == 's3':
                # strategy 3 for fmg
                segScoreMean = segImgInfo[y1:y2, x1:x2].mean()
                if segScoreMean >= segWeight:
                    newDetScore = detImgBoxs[i][4]
                else:
                    newDetScore = 0.7  # 0.7
                detImgBox = np.append(detImgBoxs[i], [segScoreMean, newDetScore])
                detImgInfo[i] = ' '.join(detImgBox.astype(np.str))
            elif strategyType == 's4':
                # strategy 3 for fmg
                segScoreMean = segImgInfo[y1:y2, x1:x2].mean()
                if detImgBoxs[i][4] > 0.8:
                    if segScoreMean >= segWeight:
                        newDetScore = detImgBoxs[i][4]
                    else:
                        newDetScore = 0.5  # 0.7
                elif detImgBoxs[i][4] > 0.7 and detImgBoxs[i][4] < 0.8:
                    if segScoreMean >= 0.209:
                        newDetScore = 1
                    else:
                        newDetScore = detImgBoxs[i][4]
                else:
                    newDetScore = detImgBoxs[i][4]

                detImgBox = np.append(detImgBoxs[i], [segScoreMean, newDetScore])
                detImgInfo[i] = ' '.join(detImgBox.astype(np.str))
    return detImgInfos


def loadResultFile(filePath):
    # load annotated image
    with open(filePath, 'r') as f:
        lines = f.readlines()
    index = 0
    imgsInfo = {}
    im_names = linecache.getlines(testImgList)
    for i in range(len(im_names)):
        im = im_names[i][:-1]
        aim = lines[index][:-1]
        assert im == aim, "image is not matched"
        index = index + 1
        roiNum = int(lines[index][:-1])
        imgsInfo[aim] = [lines[j][:-1] for j in range(index + 1, index + 1 + roiNum)]
        index = index + roiNum + 1
    return imgsInfo


def loadResultFile_voc(filePath, im_names):
    # load annotated image
    index = 0
    imgsInfo = {}
    for i, im_name in enumerate(im_names):
        im_name = im_names[i][:-1]
        imgsInfo[im_name] = parse_rec(filePath.format(im_name))
        R = [obj for obj in imgsInfo[im_name]]
        bbox = np.array([x['bbox'] for x in R])
        imgsInfo[im_name] = bbox

    return imgsInfo


def loadResultFile_300w_face(threeHus_fileDir):
    # load annotated image
    imgsInfo = {}
    for dir, subdir, files in os.walk(threeHus_fileDir):
        for file in files:
            if file.endswith('.pts'):
                im_path = os.path.join(dir, file.replace('.pts', '.png'))
                label_path = os.path.join(dir, file)
                assert os.path.exists(im_path)
                assert os.path.exists(label_path)
                results = linecache.getlines(label_path)
                results = np.array(([x[:-1].split(' ') for x in results[3:71]]), dtype=np.float)
                imgsInfo[im_path] = results
    assert len(imgsInfo) == 600
    return imgsInfo


def loadDetectFile_voc(vocDetectFilePath, im_names):
    with open(vocDetectFilePath, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = np.array([x[0] for x in splitlines])
    scores = np.array([float(x[1]) for x in splitlines])
    boxes = np.array([[float(z) for z in x[2:]] for x in splitlines])

    detImgInfos = {}
    for im_name in im_names:
        im_name = im_name[:-1]
        image_ids_index = np.where(image_ids == im_name)[0]
        if len(image_ids_index) != 0:
            im_boxes = boxes[image_ids_index]
            im_scores = scores[image_ids_index]
            im_num = len(im_scores)
            im_info = np.hstack([im_boxes, im_scores.reshape(im_num, 1)])
            detImgInfos[im_name] = im_info
        else:
            detImgInfos[im_name] = []

    return detImgInfos


def loadDetectFile_facePlus(facePlusDetectFilePath, imPaths):
    with open(facePlusDetectFilePath, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = np.array([x[0] for x in splitlines])
    scores = np.array([float(x[1]) for x in splitlines])
    boxes = np.array([[float(z) for z in x[2:6]] for x in splitlines])
    keyPoints = np.array([[float(z) for z in x[6:]] for x in splitlines])

    detImgInfos = {}
    for im_path in imPaths:
        if im_path.endswith('\n'):
            im_path = im_path[:-1]
        image_ids_index = np.where(image_ids == im_path)[0]
        if len(image_ids_index) != 0:
            im_boxes = boxes[image_ids_index]
            im_scores = scores[image_ids_index]
            im_keyPoints = keyPoints[image_ids_index]
            im_num = len(im_scores)
            im_info = np.hstack([im_boxes, im_scores.reshape(im_num, 1), im_keyPoints])
            detImgInfos[im_path] = im_info
        else:
            detImgInfos[im_path] = []

    return detImgInfos


def loadDetectFile_facePlus_v1(facePlusDetectFilePath, imPaths):
    with open(facePlusDetectFilePath, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = np.array([x[0] for x in splitlines])
    scores = np.array([float(x[1]) for x in splitlines])
    boxes = np.array([[float(z) for z in x[2:6]] for x in splitlines])
    keyPoints = np.array([[float(z) for z in x[6:-3]] for x in splitlines])
    ages = np.array([float(x[-3]) for x in splitlines])
    genders = np.array([float(x[-2]) for x in splitlines])
    ethricities = np.array([float(x[-1]) for x in splitlines])

    detImgInfos = {}
    for im_path in imPaths:
        if im_path.endswith('\n'):
            im_path = im_path[:-1]
        image_ids_index = np.where(image_ids == im_path)[0]
        if len(image_ids_index) != 0:
            im_boxes = boxes[image_ids_index]
            im_scores = scores[image_ids_index]
            im_ages = ages[image_ids_index]
            im_genders = genders[image_ids_index]
            im_ethricities = ethricities[image_ids_index]
            im_keyPoints = keyPoints[image_ids_index]
            im_num = len(im_scores)
            im_info = np.hstack([im_boxes, im_scores.reshape(im_num, 1), im_keyPoints,
                                 im_ages.reshape(im_num, 1), im_genders.reshape(im_num, 1),
                                 im_ethricities.reshape(im_num, 1)])
            detImgInfos[im_path] = im_info
        else:
            detImgInfos[im_path] = []

    return detImgInfos


def cal_ap_mp(oriImgInfos, detImgInfos, im_names, overthresh=0.5, recordIMFP=0):
    # prepare make annotated image
    oriInfos = {}
    oriDet = {}
    oriRoiSum = 0
    for i in range(len(im_names)):
        im_name = im_names[i][:-1]
        oriImgInfo = oriImgInfos[im_name]
        oriImgBoxs = [oriImgBox.split(' ') for oriImgBox in oriImgInfo]
        oriInfos[im_name] = []
        for ori_i in range(len(oriImgBoxs)):
            major_axis_radius = int(float(oriImgBoxs[ori_i][0]))
            minor_axis_radius = int(float(oriImgBoxs[ori_i][1]))
            angle = (-1) * float(oriImgBoxs[ori_i][2]) * 10
            center_x = int(float(oriImgBoxs[ori_i][3]))
            center_y = int(float(oriImgBoxs[ori_i][4]))
            oriArea = math.pi * float(oriImgBoxs[ori_i][0]) * float(oriImgBoxs[ori_i][1])
            oriInfos[im_name].append([major_axis_radius, minor_axis_radius, angle, center_x, center_y, oriArea])
        oriDet[im_name] = [False] * len(oriImgBoxs)
        oriRoiSum += len(oriImgBoxs)

    # sorted detImgInfos
    detScores = []
    detBoxes = []
    detImNames = []
    for i in range(len(im_names)):
        im_name = im_names[i][:-1]
        detScores.extend([float(detImgBox.split(' ')[-1]) for detImgBox in detImgInfos[im_name]])
        detBoxe = [np.array(detImgBox.split(' ')[:4]).astype(float) for detImgBox in detImgInfos[im_name]]
        detBoxes.extend(detBoxe)
        detNum = len(detImgInfos[im_name])
        detImNames.extend([im_name] * detNum)
    sorted_ind = np.argsort(-np.array(detScores))
    sorted_scores = (-1) * np.sort(-np.array(detScores))
    sorted_detImNames = np.array([detImNames[i] for i in sorted_ind])
    sorted_detBoxes = [detBoxes[i] for i in sorted_ind]  # detBoxes[sorted_ind, :]

    # go down dets and mark TPs and FPs
    nd = len(sorted_detImNames)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        sorted_detBox = sorted_detBoxes[d]
        det_imgName = sorted_detImNames[d]
        det_sx = int(sorted_detBox[0])
        det_sy = int(sorted_detBox[1])
        det_w = sorted_detBox[2]
        det_h = sorted_detBox[3]
        det_ex = int(det_sx + det_w)
        det_ey = int(det_sy + det_h)
        det_area = (det_w + 1.) * (det_h + 1.)
        # annoIm = annoIms[det_imgName]
        annoIm = cv2.imread(testImgDir + det_imgName + '.jpg')
        # oriImgInfo = oriImgInfos[det_imgName]
        # oriImgBoxs = [oriImgBox.split(' ') for oriImgBox in oriImgInfo]
        oriInfo = oriInfos[det_imgName]
        det_overlaps = []

        for ori_i in range(len(oriInfo)):
            [major_axis_radius, minor_axis_radius, angle, center_x, center_y, oriArea] = oriInfo[ori_i]
            cv2.ellipse(annoIm, (center_x, center_y), (minor_axis_radius, major_axis_radius), angle, 0, 360,
                        (0, 255, ori_i), -3)
            det_reg = annoIm[det_sy:det_ey + 1, det_sx:det_ex + 1, :].reshape(
                (det_ey - det_sy + 1) * (det_ex - det_sx + 1), 3)
            inters = np.sum(np.sum(det_reg == [0, 255, ori_i], 1) == 3)
            unions = oriArea + det_area - inters
            overlaps = inters / unions
            det_overlaps.append(overlaps)
        overmax = np.max(det_overlaps)
        overmax_i = np.argmax(det_overlaps)
        if overmax > overthresh:  # prior to class score
            if not oriDet[det_imgName][overmax_i]:
                tp[d] = 1.
                oriDet[det_imgName][overmax_i] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # record IM FP
    if recordIMFP:
        record_IM_FP(fp, sorted_scores, sorted_detImNames)
        record_IM_TP(tp, sorted_scores, sorted_detImNames)
    # compute precision recall
    fp_sum = sum(fp)
    tp_sum = sum(tp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(oriRoiSum)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, False)  # use_07_metric
    # print result
    # print('oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f' % (oriRoiSum, nd, tp_sum, fp_sum, ap))
    # return rec, prec, ap
    return [ap, fp_sum, tp_sum, fp, tp, rec, prec, oriRoiSum]


def cal_ap_mp_voc(oriImgInfos, vocDetectFilePath, im_names, overthresh=0.5, recordIMFP=0):
    # prepare make annotated image
    oriInfos = {}
    oriDet = {}
    oriRoiSum = 0
    for i in range(len(im_names)):
        im_name = im_names[i][:-1]
        oriImgBoxs = oriImgInfos[im_name]
        oriDet[im_name] = [False] * len(oriImgBoxs)
        oriInfos[im_name] = oriImgBoxs
        oriRoiSum += len(oriImgBoxs)

    with open(vocDetectFilePath, 'r') as f:
        lines = f.readlines()

    # when nothing to detect
    if len(lines) == 0:
        ret = [0] * 8
        ret.append('oriAnnNum:0 detNum:0 det_TP:0 det_FP:0 det_AP:0.0')
        print ret[-1]
        return ret

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = (-1) * np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = oriInfos[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R.astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > overthresh:
            # if not R['difficult'][jmax]:
            if not oriDet[image_ids[d]][jmax]:
                tp[d] = 1.
                oriDet[image_ids[d]][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # plot FP or TP
    if recordIMFP:
        plot_IM_FPTP(fp, tp, sorted_scores)
    # compute precision recall
    fp_sum = sum(fp)
    tp_sum = sum(tp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(oriRoiSum)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, False)  # use_07_metric

    print('oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f' % (oriRoiSum, nd, tp_sum, fp_sum, ap))
    str = 'oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f' % (oriRoiSum, nd, tp_sum, fp_sum, ap)
    return [ap, fp_sum, tp_sum, fp, tp, rec, prec, oriRoiSum, str]


def cal_ap_mp_facePlus(faceImages, DetectFilePath, overthresh=0.5, recordIMFP=0):
    # prepare make annotated image
    oriInfos = {}
    oriDet = {}
    oriRoiSum = 0
    oriRoiKpSum = 0
    for faceImage in faceImages:
        oriImgBoxs = faceImage.get_bboxes('xyxy')
        NoneValue = np.zeros((68, 2))
        NoneValue.fill(-1)
        oriImgKeyPoints = faceImage.get_keypoints()
        oriNoneKpNum = oriImgKeyPoints.count(None)
        # transform None into NoneValue(-1)
        oriImgKeyPoints = np.array(map(lambda x : NoneValue if x is None else x, oriImgKeyPoints)).reshape(-1, 136)
        oriImgInfos = np.hstack((oriImgBoxs, oriImgKeyPoints))
        oriDet[faceImage.image_path] = [False] * len(oriImgBoxs)
        oriInfos[faceImage.image_path] = oriImgInfos
        oriRoiSum += len(oriImgBoxs)
        oriRoiKpSum += len(oriImgBoxs) - oriNoneKpNum

    with open(DetectFilePath, 'r') as f:
        lines = f.readlines()

    # when nothing to detect
    if len(lines) == 0:
        ret = [0] * 10
        ret.append('oriAnnNum:0 detNum:0 det_TP:0 det_FP:0 det_AP:0.0 det_AVE_KP_CER:0.0')
        print ret[-1]
        return ret

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    detInfos = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    detInfos = detInfos[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # compute other metrics
    kp_cer = np.empty(nd, dtype=np.float)
    kp_cer.fill(-1)
    for d in range(nd):
        Regions = oriInfos[image_ids[d]][:, :4]
        KeyPoints = oriInfos[image_ids[d]][:, 4:]
        detInfo = detInfos[d, :].astype(float)
        ovmax = -np.inf
        BBGT = Regions.astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], detInfo[0])
            iymin = np.maximum(BBGT[:, 1], detInfo[1])
            ixmax = np.minimum(BBGT[:, 2], detInfo[2])
            iymax = np.minimum(BBGT[:, 3], detInfo[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((detInfo[2] - detInfo[0] + 1.) * (detInfo[3] - detInfo[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > overthresh:
            # if not R['difficult'][jmax]:
            if not oriDet[image_ids[d]][jmax]:
                tp[d] = 1.
                oriDet[image_ids[d]][jmax] = 1
                # compute kp cer
                if not sum(KeyPoints[jmax, :] == -1) == 136:
                    oriKeyPoint = KeyPoints[jmax, :].reshape(68, 2)
                    detKeyPoint = detInfo[4:].reshape(68, 2)
                    interocular_distance = np.linalg.norm(oriKeyPoint[36, :] - oriKeyPoint[45, :])
                    cers = 0
                    for p in range(68):
                        cers = cers + np.linalg.norm(detKeyPoint[p, :] - oriKeyPoint[p, :])
                    kp_cer[d] = cers / (68 * interocular_distance)
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp_sum = sum(fp)
    tp_sum = sum(tp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(oriRoiSum)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, False)  # use_07_metric

    # compute average of cers for only detected keyPoints
    all_kp_cer = kp_cer[np.where(kp_cer != -1)]
    ave_kp_cer = sum(all_kp_cer) / len(all_kp_cer)
    cers_list = _compute_cer_format([all_kp_cer], oriRoiKpSum)

    print('oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f det_AVE_KP_CER:%f'
          % (oriRoiSum, nd, tp_sum, fp_sum, ap, ave_kp_cer))
    str = 'oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f det_AVE_KP_CER:%f' \
          % (oriRoiSum, nd, tp_sum, fp_sum, ap, ave_kp_cer)
    return [ap, fp_sum, tp_sum, fp, tp, rec, prec, oriRoiSum, ave_kp_cer, cers_list, str]


def cal_ap_mp_facePlus_v1(faceImages, DetectFilePath, overthresh=0.5, recordIMFP=0):
    # prepare make annotated image
    oriInfos = {}
    oriDet = {}
    oriRoiSum = 0
    oriRoiKpSum = 0
    oriRoiAgeSum = 0
    oriRoiGenderSum = 0
    oriRoiEthnicitySum = 0

    for faceImage in faceImages:
        oriImgBoxs = faceImage.get_bboxes('xyxy')
        NoneValue = np.zeros((68, 2))
        NoneValue.fill(-1)
        oriImgKeyPoints = faceImage.get_keypoints()
        oriImgAges = faceImage.get_ages()
        oriImgGenders = faceImage.get_genders()
        oriImgEthnicities = faceImage.get_ethnicity()

        oriNoneKpNum = oriImgKeyPoints.count(None)
        oriNoneAgeNum = oriImgAges.count(None)
        oriNoneGenderNum = oriImgGenders.count(None)
        oriNoneEthnicityNum = oriImgEthnicities.count(None)
        # transform None into NoneValue(-1)
        gender_map = {'Male': 0, 'Female': 1}
        ethnicity_map = {'White': 0, 'Asian': 1, 'Black': 2}

        oriImgKeyPoints = np.array(map(lambda x: NoneValue if x is None else x, oriImgKeyPoints)).reshape(-1, 136)
        oriImgAges = np.array(map(lambda a: -1 if a is None else int(a), oriImgAges)).reshape(-1, 1)
        oriImgGenders = np.array(map(lambda g: -1 if g is None else gender_map[g], oriImgGenders)).reshape(-1, 1)
        oriImgEthnicities = np.array(map(lambda e: -1 if e is None else ethnicity_map[e], oriImgEthnicities)).reshape(-1, 1)

        oriImgInfos = np.hstack((oriImgBoxs, oriImgKeyPoints, oriImgAges, oriImgGenders, oriImgEthnicities))
        oriDet[faceImage.image_path] = [False] * len(oriImgBoxs)
        oriInfos[faceImage.image_path] = oriImgInfos
        oriRoiSum += len(oriImgBoxs)
        oriRoiKpSum += len(oriImgBoxs) - oriNoneKpNum
        oriRoiAgeSum += len(oriImgBoxs) - oriNoneAgeNum
        oriRoiGenderSum += len(oriImgBoxs) - oriNoneGenderNum
        oriRoiEthnicitySum += len(oriImgBoxs) - oriNoneEthnicityNum

    with open(DetectFilePath, 'r') as f:
        lines = f.readlines()

    # when nothing to detect
    if len(lines) == 0:
        ret = [0] * 13
        ret.append('oriAnnNum:0 detNum:0 det_TP:0 det_FP:0 det_AP:0.0 det_AVE_KP_CER:0.0 '
                   'det_AGE_RMSE:0.0 det_GENDER_RMSE:0.0 det_ETHNICITY_RMSE:0.0')
        print ret[-1]
        return ret

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    detInfos = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    detInfos = detInfos[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # compute other metrics
    kp_cer = np.empty(nd, dtype=np.float)
    age_ee = np.empty(nd, dtype=np.float)
    gender_ee = np.empty(nd, dtype=np.float)
    ethnicity_ee = np.empty(nd, dtype=np.float)
    kp_cer.fill(-1)
    age_ee.fill(-1)
    gender_ee.fill(-1)
    ethnicity_ee.fill(-1)

    for d in range(nd):
        Regions = oriInfos[image_ids[d]][:, :4]
        KeyPoints = oriInfos[image_ids[d]][:, 4:-3]
        Ages = oriInfos[image_ids[d]][:, -3]
        Genders = oriInfos[image_ids[d]][:, -2]
        Ethnicities = oriInfos[image_ids[d]][:, -1]
        detInfo = detInfos[d, :].astype(float)
        ovmax = -np.inf
        BBGT = Regions.astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], detInfo[0])
            iymin = np.maximum(BBGT[:, 1], detInfo[1])
            ixmax = np.minimum(BBGT[:, 2], detInfo[2])
            iymax = np.minimum(BBGT[:, 3], detInfo[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((detInfo[2] - detInfo[0] + 1.) * (detInfo[3] - detInfo[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > overthresh:
            # if not R['difficult'][jmax]:
            if not oriDet[image_ids[d]][jmax]:
                tp[d] = 1.
                oriDet[image_ids[d]][jmax] = 1
                # compute kp cer
                if not sum(KeyPoints[jmax, :] == -1) == 136:
                    oriKeyPoint = KeyPoints[jmax, :].reshape(68, 2)
                    detKeyPoint = detInfo[4:-3].reshape(68, 2)
                    interocular_distance = np.linalg.norm(oriKeyPoint[36, :] - oriKeyPoint[45, :])
                    cers = 0
                    for p in range(68):
                        cers = cers + np.linalg.norm(detKeyPoint[p, :] - oriKeyPoint[p, :])
                    kp_cer[d] = cers / (68 * interocular_distance)
                # compute rmse of age, gender and ethnicity
                if Ages[jmax] != -1:
                    oriAge = Ages[jmax]
                    detAge = detInfo[-3]
                    age_ee[d] = (oriAge-detAge)**2
                if Genders[jmax] != -1:
                    oriGender = Genders[jmax]
                    detGender = detInfo[-2]
                    gender_ee[d] = (oriGender-detGender)**2
                if Ethnicities[jmax] != -1:
                    oriEthnicity = Ethnicities[jmax]
                    detEthnicity = detInfo[-1]
                    ethnicity_ee[d] = (oriEthnicity-detEthnicity)**2
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp_sum = sum(fp)
    tp_sum = sum(tp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(oriRoiSum)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, False)  # use_07_metric

    # compute average of cers for only detected keyPoints
    all_kp_cer = kp_cer[np.where(kp_cer != -1)]
    ave_kp_cer = sum(all_kp_cer) / len(all_kp_cer)
    cers_list = _compute_cer_format([all_kp_cer], oriRoiKpSum)

    # compute rmse for only detected age, gender and ethnicity
    all_age_ee = age_ee[np.where(age_ee != -1)]
    age_rmse = np.sqrt(sum(all_age_ee) / len(all_age_ee))
    all_gender_ee = gender_ee[np.where(gender_ee != -1)]
    gender_rmse = np.sqrt(sum(all_gender_ee) / len(all_gender_ee))
    all_ethnicity_ee = ethnicity_ee[np.where(ethnicity_ee != -1)]
    ethnicity_rmse = np.sqrt(sum(all_ethnicity_ee) / len(all_ethnicity_ee))

    print('oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f det_AVE_KP_CER:%f '
          'det_AGE_RMSE:%f det_GENDER_RMSE:%f det_ETHNICITY_RMSE:%f'
          % (oriRoiSum, nd, tp_sum, fp_sum, ap, ave_kp_cer, age_rmse, gender_rmse, ethnicity_rmse))
    str = 'oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f det_AVE_KP_CER:%f ' \
          'det_AGE_RMSE:%f det_GENDER_RMSE:%f det_ETHNICITY_RMSE:%f' \
          % (oriRoiSum, nd, tp_sum, fp_sum, ap, ave_kp_cer, age_rmse, gender_rmse, ethnicity_rmse)
    return [ap, fp_sum, tp_sum, fp, tp, rec, prec, oriRoiSum, ave_kp_cer, cers_list,
            age_rmse, gender_rmse, ethnicity_rmse, str]


def _compute_cer_threeHus(oriImgInfos, detImgInfos, detType='All'):
    if detType == 'Indoor':
        targetKeys = filter(lambda i: i.split('/')[-2].endswith('Indoor'), oriImgInfos.keys())
        assert len(targetKeys) == 300
    elif detType == 'Outdoor':
        targetKeys = filter(lambda i: i.split('/')[-2].endswith('Outdoor'), oriImgInfos.keys())
        assert len(targetKeys) == 300
    elif detType == 'All':
        targetKeys = oriImgInfos.keys()
        assert len(targetKeys) == 600

    num_of_images = len(targetKeys)
    error_per_image = np.empty([num_of_images, 2])
    error_per_image.fill(-1)  # set error of image not detected is -1

    for k, num_of_points in enumerate([68, 51]):
        for i, img in enumerate(targetKeys):
            if len(detImgInfos[img]) == 0:
                continue
                # raise customError('nothing face to detect')
            elif detImgInfos[img].shape[0] > 1:
                maxIndex = np.argmax(detImgInfos[img][:, 4])
                detected_points = detImgInfos[img][maxIndex, 5:].reshape(-1, 2)
            elif detImgInfos[img].shape[0] == 1:
                detected_points = detImgInfos[img][0, 5:].reshape(-1, 2)
            ground_truth_points = oriImgInfos[img]

            if num_of_points == 51:
                detected_points = detected_points[17:, :]
                ground_truth_points = ground_truth_points[17:, :]

            if num_of_points == 68:
                interocular_distance = np.linalg.norm(ground_truth_points[36, :] - ground_truth_points[45, :])
            elif num_of_points == 51:
                interocular_distance = np.linalg.norm(ground_truth_points[19, :] - ground_truth_points[28, :])

            sum = 0
            for p in range(num_of_points):
                sum = sum + np.linalg.norm(detected_points[p, :] - ground_truth_points[p, :])

            error_per_image[i, k] = sum / (num_of_points * interocular_distance)
    return error_per_image


def _compute_cer_format(cers_threeHus, cers_sum=None, start=0, end=0.3505, internal=0.0005):
    cer_keys = np.arange(start, end, internal)
    cers_threeHus_num = len(cers_threeHus)
    cer_list = np.zeros([len(cer_keys), cers_threeHus_num])

    for cers_threeHus_i in range(cers_threeHus_num):
        for cer_index, cer_key in enumerate(cer_keys):
            cer_threeHus = cers_threeHus[cers_threeHus_i]
            cersSum = len(cer_threeHus) if cers_sum == None else cers_sum
            cer_value = len(np.where((cer_threeHus <= cer_key) &
                                     (cer_threeHus != -1))[0]) / float(cersSum)
            cer_list[cer_index, cers_threeHus_i] = cer_value

    return np.hstack([cer_keys.reshape([cer_keys.shape[0], 1]), cer_list])


def cal_cer_threeHus(oriImgInfos, detImgInfos):
    indoor_cers = _compute_cer_threeHus(oriImgInfos, detImgInfos, 'Indoor')
    outdoor_cers = _compute_cer_threeHus(oriImgInfos, detImgInfos, 'Outdoor')
    all_cers = _compute_cer_threeHus(oriImgInfos, detImgInfos, 'All')

    # compute metric of cers
    valid_indoor_cers_68 = filter(lambda x: x != -1, indoor_cers[:, 0])
    valid_indoor_cers_51 = filter(lambda x: x != -1, indoor_cers[:, 1])
    valid_outdoor_cer_68 = filter(lambda x: x != -1, outdoor_cers[:, 0])
    valid_outdoor_cer_51 = filter(lambda x: x != -1, outdoor_cers[:, 1])
    valid_all_cer_68 = filter(lambda x: x != -1, all_cers[:, 0])
    valid_all_cer_51 = filter(lambda x: x != -1, all_cers[:, 1])
    indoor_cer_68 = sum(valid_indoor_cers_68) / len(valid_indoor_cers_68)
    indoor_cer_51 = sum(valid_indoor_cers_51) / len(valid_indoor_cers_51)
    outdoor_cer_68 = sum(valid_outdoor_cer_68) / len(valid_outdoor_cer_68)
    outdoor_cer_51 = sum(valid_outdoor_cer_51) / len(valid_outdoor_cer_51)
    all_cer_68 = sum(valid_all_cer_68) / len(valid_all_cer_68)
    all_cer_51 = sum(valid_all_cer_51) / len(valid_all_cer_51)

    print('indoor_cer_68:%.5f outdoor_cer_68:%.5f all_cer_68:%.5f' % (indoor_cer_68, outdoor_cer_68, all_cer_68))
    str = 'indoor_cer_68:%.5f outdoor_cer_68:%.5f all_cer_68:%.5f' % (indoor_cer_68, outdoor_cer_68, all_cer_68)

    # Bin 68_all 68_indoor 68_outdoor 51_all 51_indoor 51_outdoor
    cers_threeHus = [all_cers[:, 0], indoor_cers[:, 0], outdoor_cers[:, 0], all_cers[:, 1], indoor_cers[:, 1], outdoor_cers[:, 1]]
    cers_list = _compute_cer_format(cers_threeHus)

    return [all_cer_68, indoor_cer_68, outdoor_cer_68, cers_list, str]


def record_result(metricList, metricFilePath, tpfpFilePath, includeRPN=False):
    ap = metricList[0]
    fp_sum = metricList[1]
    tp_sum = metricList[2]
    fp = metricList[3]
    tp = metricList[4]
    rec = metricList[5]
    prec = metricList[6]
    oriRoiSum = metricList[7]
    if includeRPN:
        with open(metricFilePath, 'a') as f:
            f.write('RPN ap %.8f tp_sum %d fp_sum %d oriRoiSum %d\n' % (ap, tp_sum, fp_sum, oriRoiSum))
    else:
        with open(metricFilePath, 'w') as f:
            f.write('ap %.8f tp_sum %d fp_sum %d oriRoiSum %d\n' % (ap, tp_sum, fp_sum, oriRoiSum))
    # plot and save metric
    plt.figure(1)
    plt.xlabel('FP')
    plt.ylabel('TP rate')
    plt.xlim(xmax=2000, xmin=0)
    plt.ylim(ymax=1, ymin=0)
    plt.title('FP_TP rate Metric')
    plt.plot(fp, rec)
    plt.legend()

    # plt.show()
    plt.savefig(tpfpFilePath)
    plt.close('all')


def record_result_facePlus(metricList, metricFilePath, tpfpFilePath):
    ap = metricList[0]
    fp_sum = metricList[1]
    tp_sum = metricList[2]
    fp = metricList[3]
    tp = metricList[4]
    rec = metricList[5]
    prec = metricList[6]
    oriRoiSum = metricList[7]
    ave_kp_cer = metricList[8]
    cers_list = metricList[9]
    with open(metricFilePath, 'w') as f:
        f.write('ap %.8f tp_sum %d fp_sum %d oriRoiSum %d ave_kp_cer %.8f\n' % (ap, tp_sum, fp_sum, oriRoiSum, ave_kp_cer))
        f.write('300W Challenge 2013 Results\n')
        f.write('-----------------------------\n')
        f.write('Bin 68_all\n')
        cers_list_str = np.round(cers_list, 4).astype(np.str)
        for str in cers_list_str:
            f.write(' '.join(str)+'\n')
    # plot and save metric
    plt.figure(1)
    plt.xlabel('FP')
    plt.ylabel('TP rate')
    plt.xlim(xmax=2000, xmin=0)
    plt.ylim(ymax=1, ymin=0)
    plt.title('FP_TP rate Metric')
    plt.plot(fp, rec)
    plt.legend()

    # plt.show()
    plt.savefig(tpfpFilePath)
    plt.close('all')


def record_result_facePlus_v1(metricList, metricFilePath, tpfpFilePath):
    ap = metricList[0]
    fp_sum = metricList[1]
    tp_sum = metricList[2]
    fp = metricList[3]
    tp = metricList[4]
    rec = metricList[5]
    prec = metricList[6]
    oriRoiSum = metricList[7]
    ave_kp_cer = metricList[8]
    cers_list = metricList[9]
    age_rmse = metricList[10]
    gender_rmse = metricList[11]
    ethnicity_rmse = metricList[12]

    with open(metricFilePath, 'w') as f:
        f.write('ap %.8f tp_sum %d fp_sum %d oriRoiSum %d ave_kp_cer %.8f'
                ' age_rmse %.8f gender_rmse %.8f ethnicity_rmse %.8f\n' %
                (ap, tp_sum, fp_sum, oriRoiSum, ave_kp_cer, age_rmse, gender_rmse, ethnicity_rmse))
        f.write('300W Challenge 2013 Results\n')
        f.write('-----------------------------\n')
        f.write('Bin 68_all\n')
        cers_list_str = np.round(cers_list, 4).astype(np.str)
        for str in cers_list_str:
            f.write(' '.join(str)+'\n')
    # plot and save metric
    plt.figure(1)
    plt.xlabel('FP')
    plt.ylabel('TP rate')
    plt.xlim(xmax=2000, xmin=0)
    plt.ylim(ymax=1, ymin=0)
    plt.title('FP_TP rate Metric')
    plt.plot(fp, rec)
    plt.legend()

    # plt.show()
    plt.savefig(tpfpFilePath)
    plt.close('all')


def record_result_threeHus(metricList, metricFilePath):
    all_cer_68 = metricList[0]
    indoor_cer_68 = metricList[1]
    outdoor_cer_68 = metricList[2]
    cers_list = metricList[3]
    with open(metricFilePath, 'w') as f:
        f.write('indoor_cer_68:%.5f outdoor_cer_68:%.5f all_cer_68:%.5f\n' % (indoor_cer_68, outdoor_cer_68, all_cer_68))
        f.write('300W Challenge 2013 Results\n')
        f.write('-----------------------------\n')
        f.write('Bin 68_all 68_indoor 68_outdoor 51_all 51_indoor 51_outdoor\n')
        cers_list_str = np.round(cers_list, 4).astype(np.str)
        for str in cers_list_str:
            f.write(' '.join(str)+'\n')


def record_results(metricLists, metricFilePath, tpfpFilePath, methods):
    metricFile = open(metricFilePath, 'w')
    plt.figure(1)
    plt.xlabel('FP')
    plt.ylabel('TP rate')
    plt.xlim(xmax=2000, xmin=0)
    plt.ylim(ymax=1, ymin=0)
    plt.title('FP_TP rate Metric')
    for i, metricList in enumerate(metricLists):
        ap = metricList[0]
        fp_sum = metricList[1]
        tp_sum = metricList[2]
        fp = metricList[3]
        tp = metricList[4]
        rec = metricList[5]
        prec = metricList[6]
        oriRoiSum = metricList[7]
        metricFile.write('%s ap %.8f tp_sum %d fp_sum %d oriRoiSum %d\n' % (methods[i], ap, tp_sum, fp_sum, oriRoiSum))
        # plot and save metric
        plt.plot(fp, rec, label=methods[i])
    plt.legend()
    # plt.show()
    plt.savefig(tpfpFilePath)
    plt.close('all')
    metricFile.close()


def label2img(labels):
    R = range(0, 255, 13)
    G = range(255, 0, -13)
    B = range(0, 255, 13)
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


def visual_det_seg(detImgInfos, segmentInfos, im_names):
    for im_name in im_names:
        # Load the demo image
        im_name = im_name[:-1]
        plt.figure()
        im = cv2.imread(testImgDir + im_name + '.jpg')
        detImgInfo = detImgInfos[im_name]
        detImgBoxs = np.array([detImgBox.split(' ') for detImgBox in detImgInfo], dtype=np.float32)
        for i in range(len(detImgBoxs)):
            x1 = detImgBoxs[i][0]
            y1 = detImgBoxs[i][1]
            x2 = detImgBoxs[i][2] + detImgBoxs[i][0]
            y2 = detImgBoxs[i][3] + detImgBoxs[i][1]
            score = detImgBoxs[i][-1]
            scale = 1  # (10/(bbox[2]-bbox[0]))
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(im, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 1)
            cv2.putText(im, str(score), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 1)
        ax = plt.subplot(1, 2, 1)
        # fig, ax = plt.subplot(2, 1, 2)
        ax.imshow(im[:, :, ::-1], aspect='equal')
        ax.set_title('detImg')
        segmentInfo = segmentInfos[im_name]
        resultLabelIm = label2img(segmentInfo)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(resultLabelIm, aspect='equal')
        ax.set_title('segImg')
        plt.close('all')
        pass


def visual_result(oriImgInfos, detImgInfos, imageSaveDir):
    mk_dir(imageSaveDir)
    for index, im_name in enumerate(oriImgInfos):
        annoImgInfo = oriImgInfos[im_name]
        im = cv2.imread(testImgDir + im_name + '.jpg')
        annoImgBoxs = np.array([annoImgBox.split(' ')[:5] for annoImgBox in annoImgInfo], dtype=np.float32)
        for i in range(len(annoImgBoxs)):
            major_axis_radius = annoImgBoxs[i][0]
            minor_axis_radius = annoImgBoxs[i][1]
            angle = (-1) * annoImgBoxs[i][2] * 10
            center_x = annoImgBoxs[i][3]
            center_y = annoImgBoxs[i][4]
            cv2.ellipse(im, (center_x, center_y), (minor_axis_radius, major_axis_radius), angle, 0, 360, (0, 255, 0), 3)

        detImgInfo = detImgInfos[im_name]
        detImgBoxs = np.array([detImgBox.split(' ') for detImgBox in detImgInfo], dtype=np.float32)
        for i in range(len(detImgBoxs)):
            x1 = detImgBoxs[i][0]
            y1 = detImgBoxs[i][1]
            x2 = detImgBoxs[i][2] + detImgBoxs[i][0]
            y2 = detImgBoxs[i][3] + detImgBoxs[i][1]
            score = detImgBoxs[i][-1]
            scale = 0.8  # (10/(bbox[2]-bbox[0]))
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # cv2.putText(im, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
            cv2.putText(im, str(score), (x1, int(y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
            imfile = imageSaveDir + '/re_' + '_'.join(im_name.split('/')) + '.jpg'
            cv2.imwrite(imfile, im)
    pass


def visual_result_voc(oriImgInfos, detImgInfos, imageSaveDir, vocValImgDir):
    mk_dir(imageSaveDir)
    for index, im_name in enumerate(oriImgInfos):
        # visual annotation
        annoImgBoxs = np.array(oriImgInfos[im_name], dtype=np.float32)
        im = cv2.imread(vocValImgDir + im_name + '.jpg')
        for i in range(len(annoImgBoxs)):
            x1 = annoImgBoxs[i][0]
            y1 = annoImgBoxs[i][1]
            x2 = annoImgBoxs[i][2]
            y2 = annoImgBoxs[i][3]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # visual detection
        detImgBoxs = np.array(detImgInfos[im_name], dtype=np.float32)
        if len(detImgBoxs) == 0:
            continue
        for i in range(len(detImgBoxs)):
            x1 = detImgBoxs[i][0]
            y1 = detImgBoxs[i][1]
            x2 = detImgBoxs[i][2]
            y2 = detImgBoxs[i][3]
            score = detImgBoxs[i][-1]
            scale = 0.8  # (10/(bbox[2]-bbox[0]))
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # cv2.putText(im, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
            cv2.putText(im, str(score), (x1, int(y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
            imfile = imageSaveDir + '/re_' + '_'.join(im_name.split('/')) + '.jpg'
            cv2.imwrite(imfile, im)
    pass


def visual_result_facePlus(faceImages, detImgInfos, imageSaveDir):
    mk_dir(imageSaveDir)
    for faceImage in faceImages:
        # visual annotation
        annoImgBoxs = faceImage.get_bboxes('xyxy')
        # kps = faceImage.get_keypoints().reshape(-1, 2)
        kps = np.array(filter(lambda kp: kp is not None, faceImage.get_keypoints())).reshape(-1, 2)
        img = faceImage.get_opened_image()
        im_path = faceImage.image_path

        f = plt.figure()
        subplot = f.add_subplot(121)
        plt.imshow(img)
        draw_bbox(subplot, annoImgBoxs)
        plt.plot(kps[:, 0], kps[:, 1], 'go', ms=1.5, alpha=1)

        # visual detection
        subplot = f.add_subplot(122)
        plt.imshow(img)
        detImgInfo = np.array(detImgInfos[im_path], dtype=np.float32)
        if len(detImgInfo) == 0:
            plt.close('all')
            continue
        detImgBoxs = detImgInfo[:, 0:4]
        detImgScores = detImgInfo[:, 4]
        detImgKps = detImgInfo[:, 5:].reshape((-1, 2))
        draw_bbox(subplot, detImgBoxs)
        plt.plot(detImgKps[:, 0], detImgKps[:, 1], 'go', ms=1.5, alpha=1)
        draw_label(subplot, detImgBoxs, [detImgScores], fontcolor="red")
        imfile = imageSaveDir + '/re_' + '_'.join(im_path.split('/')[-3:])
        plt.savefig(imfile, dpi=100)
        plt.close('all')

        # cv
        # im_path = faceImage.image_path
        # im = cv2.imread(faceImage.image_path)
        # for i in range(len(annoImgBoxs)):
        #     x1 = int(annoImgBoxs[i][0])
        #     y1 = int(annoImgBoxs[i][1])
        #     x2 = int(annoImgBoxs[i][2])
        #     y2 = int(annoImgBoxs[i][3])
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # visual detection
        # detImgBoxs = np.array(detImgInfos[im_path], dtype=np.float32)
        # for i in range(len(detImgBoxs)):
        #     x1 = int(detImgBoxs[i][0])
        #     y1 = int(detImgBoxs[i][1])
        #     x2 = int(detImgBoxs[i][2])
        #     y2 = int(detImgBoxs[i][3])
        #     score = detImgBoxs[i][5]
        #     scale = 0.8  # (10/(bbox[2]-bbox[0]))
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #     # cv2.putText(im, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
        #     cv2.putText(im, str(score), (x1, int(y2+15)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
        #     imfile = imageSaveDir + '/re_' + '_'.join(im_path.split('/')[-3:]) + '.jpg'
        #     cv2.imwrite(imfile, im)
    pass


def visual_result_facePlus_v1(faceImages, detImgInfos, imageSaveDir):
    mk_dir(imageSaveDir)
    for faceImage in faceImages:
        # visual annotation
        annoImgBoxs = faceImage.get_bboxes('xyxy')
        # kps = faceImage.get_keypoints().reshape(-1, 2)
        kps = np.array(filter(lambda kp: kp is not None, faceImage.get_keypoints())).reshape(-1, 2)
        img = faceImage.get_opened_image()
        im_path = faceImage.image_path
        annoImgAges = map(lambda a: int(a) if a is not None else '', faceImage.get_ages())
        annoImgGenders = map(lambda g: g if g is not None else '', faceImage.get_genders())
        annoImgEthnicities = map(lambda e: e if e is not None else '', faceImage.get_ethnicity())

        f = plt.figure()
        subplot = f.add_subplot(121)
        plt.imshow(img)
        draw_bbox(subplot, annoImgBoxs)
        plt.plot(kps[:, 0], kps[:, 1], 'go', ms=1.5, alpha=1)
        draw_label(subplot, annoImgBoxs, [annoImgAges, annoImgGenders, annoImgEthnicities], fontcolor="red")

        # visual detection
        subplot = f.add_subplot(122)
        plt.imshow(img)
        detImgInfo = np.array(detImgInfos[im_path], dtype=np.float32)
        if len(detImgInfo) == 0:
            plt.close('all')
            continue
        detImgBoxs = detImgInfo[:, 0:4]
        detImgScores = detImgInfo[:, 4]
        detImgKps = detImgInfo[:, 5:-3].reshape((-1, 2))
        detImgAges = detImgInfo[:, -3]
        detImgGenders = detImgInfo[:, -2]
        detImgEthricities = detImgInfo[:, -1]

        labels = []
        labels.append(detImgScores)
        labels.append(_attribute_map(detImgAges, 'gt_ages'))
        labels.append(_attribute_map(detImgGenders, 'gt_genders'))
        labels.append(_attribute_map(detImgEthricities, 'gt_ethnicity'))

        draw_bbox(subplot, detImgBoxs)
        plt.plot(detImgKps[:, 0], detImgKps[:, 1], 'go', ms=1.5, alpha=1)
        draw_label(subplot, detImgBoxs, labels, fontcolor="red")
        imfile = imageSaveDir + '/re_' + '_'.join(im_path.split('/')[-3:])
        plt.savefig(imfile, dpi=100)
        plt.close('all')

        # cv
        # im_path = faceImage.image_path
        # im = cv2.imread(faceImage.image_path)
        # for i in range(len(annoImgBoxs)):
        #     x1 = int(annoImgBoxs[i][0])
        #     y1 = int(annoImgBoxs[i][1])
        #     x2 = int(annoImgBoxs[i][2])
        #     y2 = int(annoImgBoxs[i][3])
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # visual detection
        # detImgBoxs = np.array(detImgInfos[im_path], dtype=np.float32)
        # for i in range(len(detImgBoxs)):
        #     x1 = int(detImgBoxs[i][0])
        #     y1 = int(detImgBoxs[i][1])
        #     x2 = int(detImgBoxs[i][2])
        #     y2 = int(detImgBoxs[i][3])
        #     score = detImgBoxs[i][5]
        #     scale = 0.8  # (10/(bbox[2]-bbox[0]))
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #     # cv2.putText(im, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
        #     cv2.putText(im, str(score), (x1, int(y2+15)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
        #     imfile = imageSaveDir + '/re_' + '_'.join(im_path.split('/')[-3:]) + '.jpg'
        #     cv2.imwrite(imfile, im)
    pass


def _attribute_map(attrs, attrName):
    if attrName == 'gt_ages':
        attrs_str = map(lambda k: '%d' %k if k!=-1 else '', attrs) # years old
    elif attrName == 'gt_genders':
        gender_map = {0:'Male',1:'Female',-1:""}
        attrs_str = map(lambda k: gender_map[k], attrs)
    elif attrName == 'gt_ethnicity':
        ethnicity_map = {0:'White',1:'Asian',2:'Black',-1:""}
        attrs_str = map(lambda k: ethnicity_map[k], attrs)
    return attrs_str


def visual_result_300w_face(oriImgInfos, detImgInfos, imageSaveDir):
    print 'start visual..'
    mk_dir(imageSaveDir, 1)
    for index, im_path in enumerate(oriImgInfos):
        # visual annotation
        im = cv2.imread(im_path)
        f = plt.figure()
        subplot = f.add_subplot(121)
        plt.imshow(im)
        if oriImgInfos[im_path].shape[1] == 141:
            oriImgInfo = np.array(oriImgInfos[im_path], dtype=np.float32)
            oriImgBoxs = oriImgInfo[:, 0:4]
            oriImgScores = oriImgInfo[:, 4]
            oriImgKps = oriImgInfo[:, 5:].reshape((-1, 2))
            draw_bbox(subplot, oriImgBoxs)
            plt.plot(oriImgKps[:, 0], oriImgKps[:, 1], 'go', ms=1.5, alpha=1)
            draw_label(subplot, oriImgBoxs, [oriImgScores], fontcolor="red")
        else:
            kps = np.array(oriImgInfos[im_path], dtype=np.float32)
            kps = kps.reshape(-1, 2)
            plt.plot(kps[:, 0], kps[:, 1], 'go', ms=1.5, alpha=1)

        # visual detection
        subplot = f.add_subplot(122)
        plt.imshow(im)
        detImgInfo = np.array(detImgInfos[im_path], dtype=np.float32)
        if len(detImgInfo) == 0:
            plt.close('all')
            continue
        detImgBoxs = detImgInfo[:, 0:4]
        detImgScores = detImgInfo[:, 4]
        detImgKps = detImgInfo[:, 5:].reshape((-1, 2))
        draw_bbox(subplot, detImgBoxs)
        plt.plot(detImgKps[:, 0], detImgKps[:, 1], 'go', ms=1.5, alpha=1)
        draw_label(subplot, detImgBoxs, [detImgScores], fontcolor="red")
        imfile = imageSaveDir + '/re_' + '_'.join(im_path.split('/')[-3:])
        plt.savefig(imfile, dpi=100)
        plt.close('all')


def record_IM_TP(tps, sorted_scores, sorted_detImNames):
    tpIns = np.where(tps == 1)[0]
    tp_sorted_detImNames = sorted_detImNames[tpIns]
    tp_sorted_scores = sorted_scores[tpIns]
    # record result
    record_i = 0
    with open(sortedImtpFilePath, 'w') as f:
        for i, tp_sorted_detImName in enumerate(tp_sorted_detImNames):
            f.write('sorted_im %s sorted_scores %.8f\n'
                    % ('_'.join(tp_sorted_detImName.split('/')), tp_sorted_scores[i]))
            if tp_sorted_scores[i] == filterThresh:
                record_i += 1
        f.write('the sum of scores = %0.1f is %d\n' % (filterThresh, record_i))
    pass

def plot_IM_FPTP(fps, tps, sorted_scores, Threshold=0.8, internal=0.01, onlyShowFp=1):
    fpIns = np.where(fps == 1)[0]
    fp_sorted_scores = sorted_scores[fpIns]
    tpIns = np.where(tps == 1)[0]
    tp_sorted_scores = sorted_scores[tpIns]
    # plot bar for fgs
    fpsBarFile = imfpFilePath.replace('.txt', '.jpg')
    plt.figure(figsize=(10, 6))
    tpsNum = [len(np.where((tp_sorted_scores > i) & (tp_sorted_scores <= i+internal))[0]) for i in np.arange(Threshold-internal, 1, internal)]
    fpsNum = [len(np.where((fp_sorted_scores > i) & (fp_sorted_scores <= i+internal))[0]) for i in np.arange(Threshold-internal, 1, internal)]
    scoresRange = ['%s' % str(i) for i in np.arange(Threshold-internal, 1, internal)]
    xFPs = range(0, len(fpsNum))
    xTPs = [i+0.3 for i in xFPs]
    if onlyShowFp:
        # plt.yticks(np.arange(0, max(fpsNum) + 20, 20))
        plt.bar(xFPs, fpsNum, color='r', width=.3, alpha=0.4, label='FPs')
    else:
        plt.bar(xFPs, fpsNum, color='r', width=.3, alpha=0.4, label='FPs')
        plt.bar(xTPs, tpsNum, color='g', width=.3, alpha=0.6, label='TPs')
    plt.xlabel('scoresRange')
    plt.ylabel('Num')
    plt.title('FPTP-SCORE-DISTRIBUTION')
    plt.legend(loc='upper right')
    plt.xticks(range(0, len(tpsNum)), scoresRange)
    # plt.show()
    plt.grid()
    plt.savefig(fpsBarFile)
    plt.close('all')

def record_IM_FP(fps, sorted_scores, sorted_detImNames):
    fpIns = np.where(fps == 1)[0]
    fp_sorted_detImNames = sorted_detImNames[fpIns]
    fp_sorted_scores = sorted_scores[fpIns]
    unique_detImNames = np.array(list(set(fp_sorted_detImNames)))
    # sorted im by fpSum
    ims_fpSum = np.array([np.sum(fp_sorted_detImNames == im) for im in unique_detImNames])
    sorted_ims_fpSum_i = np.argsort(-ims_fpSum)
    sorted_ims_fpSum = ims_fpSum[sorted_ims_fpSum_i]
    sorted_imNames_by_fpSum = unique_detImNames[sorted_ims_fpSum_i]
    # sorted im by fpScores
    ims_first_index = np.array([np.where(fp_sorted_detImNames == im)[0][0] for im in unique_detImNames])
    ims_first_index_scores = fp_sorted_scores[ims_first_index]
    sorted_ims_first_index_i = np.argsort(ims_first_index)
    sorted_ims_first_index_scores = ims_first_index_scores[sorted_ims_first_index_i]
    sorted_imNames_by_scores = unique_detImNames[sorted_ims_first_index_i]
    # record result
    with open(imfpFilePath, 'w') as f:
        for i in range(len(unique_detImNames)):
            f.write('im_by_fpSum %s fpSum %d im_by_scores %s scores %.8f\n'
                    % ('_'.join(sorted_imNames_by_fpSum[i].split('/')), sorted_ims_fpSum[i],
                       '_'.join(sorted_imNames_by_scores[i].split('/')), sorted_ims_first_index_scores[i]))
    record_i = 0
    with open(sortedImfpFilePath, 'w') as f:
        for i, fp_sorted_detImName in enumerate(fp_sorted_detImNames):
            f.write('sorted_im %s sorted_scores %.8f\n'
                    % ('_'.join(fp_sorted_detImName.split('/')), fp_sorted_scores[i]))
            if fp_sorted_scores[i] == filterThresh:
                record_i += 1
        f.write('the sum of scores = %0.1f is %d\n' % (filterThresh, record_i))
    pass


def gather_result():
    methods = ['VGG16_rpn_stage3_iter_80000_v3_c8_n3',
               'VGG16_rpn_stage4_iter_80000_v3_c8_n3',  # 0.06/7
               'VGG16_faster_rcnn_end2end_iter_90000_v2_c8_n3',  # 1.11
               'VGG16_faster_rcnn_end2end_iter_90000_v2_c8_n3_scale2',  # 0.08
               'ResNet-50_rfcn_end2end_ohem_iter_70000_v2_c8_n3',  # 0.08
               'ResNet-50_rfcn_end2end_ohem_iter_70000_v2_c8_n3_scale2',
               'VGG16_rpn_v1_stage1_iter_80000_v3_c5_n3_s7_s3_a',
               'VGG16_rpn_v1_stage1_iter_80000_v3_c8_n3_s7_s3'
               ]  # 0.03
    save_dir = 'output/test_result'
    save_dir = os.path.join(cfg.ROOT_DIR, save_dir)
    Metric_file = os.path.join(save_dir, 'All_Metric.txt')
    TPFP_file = os.path.join(save_dir, 'All_TPFP.jpg')
    im_names = linecache.getlines(testImgList)
    oriImgInfos = loadResultFile(annoFilePath)
    metrics = []
    for method in methods:
        detectFilePath = save_dir + '/' + method + '.txt'
        # load detect file and compute metric
        detImgInfos = loadResultFile(detectFilePath)
        metric = cal_ap_mp(oriImgInfos, detImgInfos, im_names)
        metrics.append(metric)
    record_results(metrics, Metric_file, TPFP_file, methods)
    print 'done'


def strategy_bag():
    testImages = ['2002_07_19_big_img_255', '2002_07_19_big_img_300',
                  '2002_07_19_big_img_372', '2002_07_19_big_img_392',
                  '2002_07_19_big_img_463', '2002_07_19_big_img_630']


def filter_adapter(img):
    im_names = linecache.getlines(filterImgList)
    filterImgs = [im_name[:-1] for im_name in im_names]
    img = img.replace('/', '_')
    if img in filterImgs:
        return 1
    else:
        return 0


def loadROCAndPlot(RocFilesDir, oriRoiSum=5171):
    plt.figure(1)
    plt.xlabel('FP')
    plt.ylabel('TP rate')
    plt.xlim(xmax=2000, xmin=0)
    plt.ylim(ymax=1, ymin=0)
    plt.title('FP_TP rate Metric')
    for dir, subdir, files in os.walk(RocFilesDir):
        for file in files:
            results = linecache.getlines(os.path.join(dir, file))
            results = np.array([result.split(' ')[0:2] for result in results])
            results_FP = results[:, 1].astype(np.int)
            results_TPr = results[:, 0].astype(np.float32)

            # fill FPs until 2000
            thresh = 2000
            if results_FP[0] < thresh:
                # list =range(thresh, results_FP[0], -1)
                # results_FP = np.hstack([list, results_FP])
                # results_TPr = np.hstack([[results_TPr[0]]*len(list), results_TPr])
                results_FP = np.hstack([thresh, results_FP])
                results_TPr = np.hstack([[results_TPr[0]], results_TPr])
            elif results_FP[0] > thresh:
                i = 0
                head = results_FP[i]
                while head > thresh:
                    i += 1
                    head = results_FP[i]
                if head != thresh:
                    results_FP = np.hstack([thresh, results_FP[i:]])
                    results_TPr = np.hstack([results_TPr[i], results_TPr[i:]])
                else:
                    results_FP = results_FP[i:]
                    results_TPr = results_TPr[i:]

            sorted_result_ind = np.argsort(results_TPr)
            results_FP = results_FP[sorted_result_ind]
            results_TPr = results_TPr[sorted_result_ind]
            # oriRoiSum = results_TPr.shape[0]
            # rec = tp / float(oriRoiSum)

            fp = results_FP
            tp = results_TPr * float(oriRoiSum)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(results_TPr, prec, False)  # use_07_metric
            plt.plot(results_FP, results_TPr, label=file.split('.')[0] + ": %f" % ap)
            plt.legend()
            plt.show()
    # plt.savefig(tpfpFilePath)
    plt.close('all')


def mk_dirs(saveDir, subDirs):
    objectDirs = []
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    else:
        shutil.rmtree(saveDir)
        os.makedirs(saveDir)
    for subDir in subDirs:
        objectDir = os.path.join(saveDir, subDir[0][0])
        os.makedirs(objectDir)
        objectDirs.append(objectDir)
    return objectDirs


def mk_dir(saveDir, cover=1):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    elif cover:
        shutil.rmtree(saveDir)
        os.makedirs(saveDir)


def det_model_wider(prototxt, modelPath, modelType, matFile, saveDir,
                    CONF_THRESH=0.8, NMS_THRESH=0.3):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    # load mat
    data = sio.loadmat(matFile)
    event_list = data['event_list']
    file_list = data['file_list']
    objectDirs = mk_dirs(saveDir, event_list)
    for objectDir, event, files in zip(objectDirs, event_list, file_list):
        event = event[0][0]
        for file in files[0]:
            file = file[0][0]
            imPath = os.path.join(widerValDir, event, file + '.jpg')
            detectFilePath = os.path.join(objectDir, file + '.txt')
            resultList = open(detectFilePath, 'w')
            im = cv2.imread(imPath)
            # Detect all object classes and regress object bounds
            if modelType == 'rpn':
                boxes, scores = rpn.generate.im_proposals(net, im)
            else:
                scores, boxes = im_detect(net, im)
            for cls_ind, cls in enumerate(CLASSES[1:]):
                if modelType != 'rpn':
                    cls_ind += 1  # because we skipped background
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                # record result
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                title_name = event + '/' + file + '.jpg'
                resultList.write(title_name + '\n')
                resultList.write(str(len(inds)) + '\n')
                if len(inds) == 0:
                    continue
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    resultList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                     str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
            resultList.close()


def det_model_wider_pyramid_t(prototxt, modelPath, modelType, matFile, saveDir,
                            CONF_THRESH=0.8, NMS_THRESH=0.3, scales=[0]):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    # load mat
    data = sio.loadmat(matFile)
    event_list = data['event_list']
    file_list = data['file_list']
    base_scales = 600
    base_max_size = 1000
    objectDirs = mk_dirs(saveDir, event_list)
    for objectDir, event, files in zip(objectDirs, event_list, file_list):
        event = event[0][0]
        for file in files[0]:
            file = file[0][0]
            imPath = os.path.join(widerValDir, event, file + '.jpg')
            detectFilePath = os.path.join(objectDir, file + '.txt')
            resultList = open(detectFilePath, 'w')
            im = cv2.imread(imPath)
            # Detect all object classes and regress object bounds
            boxesList = []
            scoresList = []
            for scale in scales:
                if modelType == 'rpn':
                    boxes, scores = rpn.generate.im_proposals(net, im)
                else:
                    cfg.TEST.SCALES = [base_scales + scale]  # 600 800 1000 1200 1400
                    cfg.TEST.MAX_SIZE = base_max_size + scale
                    scores, boxes = im_detect(net, im.copy())

                    if cfg.PYRAMID_DETECT_DEBUG:
                        startIndex = 0 + cfg.PYRAMID_DETECT_NUM * cfg.PYRAMID_DETECT_INDEX
                        endIndex = startIndex + cfg.PYRAMID_DETECT_NUM
                        scores = scores[startIndex:endIndex]
                        boxes = boxes[startIndex:endIndex]

                    boxesList.append(boxes.copy())
                    scoresList.append(scores.copy())

            for cls_ind, cls in enumerate(CLASSES[1:]):
                # plt.figure(3)
                detsList = []
                if modelType != 'rpn':
                    cls_ind += 1  # because we skipped background
                for boxes, scores, index in zip(boxesList, scoresList, range(len(boxesList))):
                    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                    cls_scores = scores[:, cls_ind]
                    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, NMS_THRESH)
                    dets = dets[keep, :]
                    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                    # show image
                    ax = plt.subplot(2, 2, index+1)
                    newIm = showImg(im.copy(), dets[inds, :])
                    ax.imshow(newIm[:, :, ::-1], aspect='equal')
                    ax.set_title('sum of scale_%d: %d' % (scales[index], len(inds)))

                    # print 'num of dets where CONF_THRESH >= %.2f is %d' % (CONF_THRESH, len(inds))
                    if detsList == []:
                        detsList = dets[inds, :].copy()
                    else:
                        detsList = np.vstack((detsList, dets[inds, :]))

                # dets with different scale do nms
                keep = nms(detsList, NMS_THRESH)
                dets = detsList[keep, :]

                # inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                # print 'num of dets where CONF_THRESH >= %.2f is %d' % (CONF_THRESH, len(inds))

                # record result
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                # show image
                # ax = plt.subplot(2, 2, 4)
                # newIm = showImg(im, dets[inds, :])
                # ax.imshow(newIm[:, :, ::-1], aspect='equal')
                # ax.set_title('sum of scale_all: %d' % len(inds))

                title_name = event + '/' + file + '.jpg'
                resultList.write(title_name + '\n')
                resultList.write(str(len(inds)) + '\n')
                if len(inds) == 0:
                    continue
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    resultList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                     str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
            resultList.close()

def det_model_wider_pyramid(prototxt, modelPath, modelType, matFile, saveDir,
                            CONF_THRESH=0.8, NMS_THRESH=0.3, scales=[0]):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    # load mat
    data = sio.loadmat(matFile)
    event_list = data['event_list']
    file_list = data['file_list']
    base_scales = 600
    base_max_size = 1000
    objectDirs = mk_dirs(saveDir, event_list)
    for objectDir, event, files in zip(objectDirs, event_list, file_list):
        event = event[0][0]
        for file in files[0]:
            file = file[0][0]
            imPath = os.path.join(widerValDir, event, file + '.jpg')
            detectFilePath = os.path.join(objectDir, file + '.txt')
            resultList = open(detectFilePath, 'w')
            im = cv2.imread(imPath)
            # Detect all object classes and regress object bounds
            boxesList = []
            scoresList = []
            for scale in scales:
                if modelType == 'rpn':
                    boxes, scores = rpn.generate.im_proposals(net, im)
                else:
                    cfg.TEST.SCALES = [base_scales * scale]  # 600 800 1000 1200 1400
                    cfg.TEST.MAX_SIZE = base_max_size * scale
                    scores, boxes = im_detect(net, im.copy())

                    if cfg.PYRAMID_DETECT_DEBUG:
                        startIndex = 0 + cfg.PYRAMID_DETECT_NUM * cfg.PYRAMID_DETECT_INDEX
                        endIndex = startIndex + cfg.PYRAMID_DETECT_NUM
                        scores = scores[startIndex:endIndex]
                        boxes = boxes[startIndex:endIndex]

                    boxesList.append(boxes.copy())
                    scoresList.append(scores.copy())

            for cls_ind, cls in enumerate(CLASSES[1:]):
                # plt.figure(3)
                detsList = []
                if modelType != 'rpn':
                    cls_ind += 1  # because we skipped background
                for boxes, scores, index in zip(boxesList, scoresList, range(len(boxesList))):
                    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                    cls_scores = scores[:, cls_ind]
                    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, NMS_THRESH)
                    dets = dets[keep, :]
                    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                    # show image
                    ax = plt.subplot(2, 2, index+1)
                    newIm = showImg(im.copy(), dets[inds, :])
                    ax.imshow(newIm[:, :, ::-1], aspect='equal')
                    ax.set_title('sum of scale_%d: %d' % (scales[index], len(inds)))

                    # print 'num of dets where CONF_THRESH >= %.2f is %d' % (CONF_THRESH, len(inds))
                    if detsList == []:
                        detsList = dets[inds, :].copy()
                    else:
                        detsList = np.vstack((detsList, dets[inds, :]))

                # dets with different scale do nms
                keep = nms(detsList, NMS_THRESH)
                dets = detsList[keep, :]

                # inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                # print 'num of dets where CONF_THRESH >= %.2f is %d' % (CONF_THRESH, len(inds))

                # record result
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                # show image
                # ax = plt.subplot(2, 2, 4)
                # newIm = showImg(im, dets[inds, :])
                # ax.imshow(newIm[:, :, ::-1], aspect='equal')
                # ax.set_title('sum of scale_all: %d' % len(inds))

                title_name = event + '/' + file + '.jpg'
                resultList.write(title_name + '\n')
                resultList.write(str(len(inds)) + '\n')
                if len(inds) == 0:
                    continue
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    resultList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                     str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
            resultList.close()

def det_model_facePlus_pyramid(prototxt, modelPath, modelType, imPaths, DetectFilePath,
                               CONF_THRESH=0.8, NMS_THRESH=0.3, scales=[0]):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    resultList = open(DetectFilePath, 'w')
    base_scales = 600
    base_max_size = 1000
    for im_path in imPaths:
        # Load the demo image
        if im_path.endswith('\n'):
            im_path = im_path[:-1]

        im = cv2.imread(im_path)

        # Detect all object classes and regress object bounds
        boxesList = []
        scoresList = []
        keyPointsList = []
        for scale in scales:
            if modelType == 'rpn':
                boxes, scores = rpn.generate.im_proposals(net, im)
            else:
                cfg.TEST.SCALES = [base_scales + scale]  # 600 800 1000 1200 1400
                cfg.TEST.MAX_SIZE = base_max_size + scale
                scores, boxes, keyPoints = im_detect_facePlus(net, im.copy())

                if cfg.PYRAMID_DETECT_DEBUG:
                    startIndex = 0 + cfg.PYRAMID_DETECT_NUM * cfg.PYRAMID_DETECT_INDEX
                    endIndex = startIndex + cfg.PYRAMID_DETECT_NUM - 1
                    scores = scores[startIndex:endIndex]
                    boxes = boxes[startIndex:endIndex]
                    keyPoints = keyPoints[startIndex:endIndex]

            boxesList.append(boxes.copy())
            scoresList.append(scores.copy())
            keyPointsList.append(keyPoints.copy())

        for cls_ind, cls in enumerate(CLASSES[1:]):
            detsList = []
            otherdetsList = []
            if modelType != 'rpn':
                cls_ind += 1  # because we skipped background
            for boxes, scores, keyPoints in zip(boxesList, scoresList, keyPointsList):
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                # keyPoints_num = keyPoints.shape[1] / 2
                # cls_keyPoints = keyPoints[:, keyPoints_num * cls_ind:keyPoints_num * (cls_ind + 1)]
                keyPoints_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
                cls_keyPoints = keyPoints
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                det_keyPoints = cls_keyPoints[keep, :]

                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                if detsList == []:
                    detsList = dets[inds, :].copy()
                    otherdetsList = det_keyPoints[inds, :].copy()
                else:
                    detsList = np.vstack((detsList, dets[inds, :]))
                    otherdetsList = np.vstack((otherdetsList, det_keyPoints[inds, :]))

            # dets with different scale do nms
            keep = nms(detsList, NMS_THRESH)
            dets = detsList[keep, :]
            det_keyPoints = otherdetsList[keep, :]

            # record result
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                keyPoints = det_keyPoints[i, :keyPoints_num]
                keyPoints_str = ['%.1f' % key for key in keyPoints]
                # resultList.write(image_name+'\n')
                # resultList.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2]-bbox[0])+' '+
                #                  str(bbox[3]-bbox[1])+' '+str(score)+'\n')
                resultList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:s}\n'.
                                 format(im_path, score,
                                        bbox[0] + 1, bbox[1] + 1,
                                        bbox[2] + 1, bbox[3] + 1,
                                        ' '.join(keyPoints_str)))
    resultList.close()


def det_model_voc_pyramid_t(prototxt, modelPath, modelType, im_names, vocDetectFilePath,
                          vocValImgDir, CONF_THRESH=0.8, NMS_THRESH=0.3, scales=[0],
                          testingDB='wider',includeRPN=0):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    resultList = open(vocDetectFilePath, 'w')
    if includeRPN:
        resultRPNList = open(vocDetectFilePath.replace('.txt', '_RPN.txt'), 'w')
    base_scales = 600
    base_max_size = 1000
    for im_name in im_names:
        # Load the demo image
        image_name = im_name[:-1]

        im_path = os.path.join(vocValImgDir, image_name + '.jpg')
        im = cv2.imread(im_path)

        # Detect all object classes and regress object bounds
        boxesList = []
        scoresList = []
        boxesRPNList = []
        scoresRPNList = []
        for scale in scales:
            if modelType == 'rpn':
                boxes, scores = rpn.generate.im_proposals(net, im)
            else:
                cfg.TEST.SCALES = [base_scales + scale]  # 600 800 1000 1200 1400
                cfg.TEST.MAX_SIZE = base_max_size + scale

                if includeRPN:
                    scores, boxes, scoresRPN, boxesRPN = im_detect(net, im.copy(), includeRPN=True)
                else:
                    scores, boxes = im_detect(net, im.copy())

                if cfg.PYRAMID_DETECT_DEBUG:
                    startIndex = 0 + cfg.PYRAMID_DETECT_NUM * cfg.PYRAMID_DETECT_INDEX
                    endIndex = startIndex + cfg.PYRAMID_DETECT_NUM - 1
                    scores = scores[startIndex:endIndex]
                    boxes = boxes[startIndex:endIndex]

            boxesList.append(boxes.copy())
            scoresList.append(scores.copy())
            if includeRPN:
                boxesRPNList.append(boxesRPN.copy())
                scoresRPNList.append(scoresRPN.copy())

        for cls_ind, cls in enumerate(CLASSES[1:]):
            detsList = []
            if modelType != 'rpn':
                cls_ind += 1  # because we skipped background
            for boxes, scores in zip(boxesList, scoresList):
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                if detsList == []:
                    detsList = dets[inds, :].copy()
                else:
                    detsList = np.vstack((detsList, dets[inds, :]))

            # dets with different scale do nms
            keep = nms(detsList, NMS_THRESH)
            dets = detsList[keep, :]

            # record result
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                if testingDB == 'FDDB':
                    resultList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                     str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
                elif testingDB == 'wider':
                    resultList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                     format(image_name, score,
                                            bbox[0] + 1, bbox[1] + 1,
                                            bbox[2] + 1, bbox[3] + 1))
                else:
                    raise customError('testingDB is invalid')

        if includeRPN:
            for cls_ind, cls in enumerate(CLASSES[1:]):
                detsList = []
                for boxes, scores in zip(boxesRPNList, scoresRPNList):
                    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                    cls_scores = scores[:, cls_ind]
                    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, NMS_THRESH)
                    dets = dets[keep, :]
                    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                    if detsList == []:
                        detsList = dets[inds, :].copy()
                    else:
                        detsList = np.vstack((detsList, dets[inds, :]))

                # dets with different scale do nms
                keep = nms(detsList, NMS_THRESH)
                dets = detsList[keep, :]

                # record result
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                if len(inds) == 0:
                    continue
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    if testingDB == 'FDDB':
                        resultRPNList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                         str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
                    elif testingDB == 'wider':
                        resultRPNList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                         format(image_name, score,
                                                bbox[0] + 1, bbox[1] + 1,
                                                bbox[2] + 1, bbox[3] + 1))
                    else:
                        raise customError('testingDB is invalid')

    resultList.close()
    if includeRPN:
        resultRPNList.close()

def det_model_voc_pyramid(prototxt, modelPath, modelType, im_names, vocDetectFilePath,
                          vocValImgDir, CONF_THRESH=0.8, NMS_THRESH=0.3, scales=[0],
                          testingDB='wider',includeRPN=0):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    resultList = open(vocDetectFilePath, 'w')
    if includeRPN:
        resultRPNList = open(vocDetectFilePath.replace('.txt', '_RPN.txt'), 'w')
    base_scales = 600
    base_max_size = 1000
    for im_name in im_names:
        # Load the demo image
        image_name = im_name[:-1]

        im_path = os.path.join(vocValImgDir, image_name + '.jpg')
        im = cv2.imread(im_path)

        # Detect all object classes and regress object bounds
        boxesList = []
        scoresList = []
        boxesRPNList = []
        scoresRPNList = []
        for scale in scales:
            if modelType == 'rpn':
                boxes, scores = rpn.generate.im_proposals(net, im)
            else:
                cfg.TEST.SCALES = [base_scales * scale]  # 600 800 1000 1200 1400
                cfg.TEST.MAX_SIZE = base_max_size * scale

                if includeRPN:
                    scores, boxes, scoresRPN, boxesRPN = im_detect(net, im.copy(), includeRPN=True)
                else:
                    scores, boxes = im_detect(net, im.copy())

                if cfg.PYRAMID_DETECT_DEBUG:
                    startIndex = 0 + cfg.PYRAMID_DETECT_NUM * cfg.PYRAMID_DETECT_INDEX
                    endIndex = startIndex + cfg.PYRAMID_DETECT_NUM - 1
                    scores = scores[startIndex:endIndex]
                    boxes = boxes[startIndex:endIndex]

            boxesList.append(boxes.copy())
            scoresList.append(scores.copy())
            if includeRPN:
                boxesRPNList.append(boxesRPN.copy())
                scoresRPNList.append(scoresRPN.copy())

        for cls_ind, cls in enumerate(CLASSES[1:]):
            detsList = []
            if modelType != 'rpn':
                cls_ind += 1  # because we skipped background
            for boxes, scores in zip(boxesList, scoresList):
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                if detsList == []:
                    detsList = dets[inds, :].copy()
                else:
                    detsList = np.vstack((detsList, dets[inds, :]))

            # dets with different scale do nms
            keep = nms(detsList, NMS_THRESH)
            dets = detsList[keep, :]

            # record result
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                if testingDB == 'FDDB':
                    resultList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                     str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
                elif testingDB == 'wider':
                    resultList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                     format(image_name, score,
                                            bbox[0] + 1, bbox[1] + 1,
                                            bbox[2] + 1, bbox[3] + 1))
                else:
                    raise customError('testingDB is invalid')

        if includeRPN:
            for cls_ind, cls in enumerate(CLASSES[1:]):
                detsList = []
                for boxes, scores in zip(boxesRPNList, scoresRPNList):
                    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                    cls_scores = scores[:, cls_ind]
                    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, NMS_THRESH)
                    dets = dets[keep, :]
                    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                    if detsList == []:
                        detsList = dets[inds, :].copy()
                    else:
                        detsList = np.vstack((detsList, dets[inds, :]))

                # dets with different scale do nms
                keep = nms(detsList, NMS_THRESH)
                dets = detsList[keep, :]

                # record result
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                if len(inds) == 0:
                    continue
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    if testingDB == 'FDDB':
                        resultRPNList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                         str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
                    elif testingDB == 'wider':
                        resultRPNList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                         format(image_name, score,
                                                bbox[0] + 1, bbox[1] + 1,
                                                bbox[2] + 1, bbox[3] + 1))
                    else:
                        raise customError('testingDB is invalid')

    resultList.close()
    if includeRPN:
        resultRPNList.close()


def det_model_facePlus_v1_pyramid(prototxt, modelPath, modelType, imPaths, DetectFilePath,
                               CONF_THRESH=0.8, NMS_THRESH=0.3, scales=[0]):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    resultList = open(DetectFilePath, 'w')
    base_scales = 600
    base_max_size = 1000
    for im_path in imPaths:
        # Load the demo image
        if im_path.endswith('\n'):
            im_path = im_path[:-1]

        im = cv2.imread(im_path)

        # Detect all object classes and regress object bounds
        boxesList = []
        scoresList = []
        keyPointsList = []
        agesList = []
        gendersList = []
        ethricitiesList = []
        for scale in scales:
            if modelType == 'rpn':
                boxes, scores = rpn.generate.im_proposals(net, im)
            else:
                cfg.TEST.SCALES = [base_scales + scale]  # 600 800 1000 1200 1400
                cfg.TEST.MAX_SIZE = base_max_size + scale
                scores, boxes, keyPoints, age_pros, gender_pros, etnricity_pros = im_detect_facePlus_v1(net, im.copy())

                if cfg.PYRAMID_DETECT_DEBUG:
                    startIndex = 0 + cfg.PYRAMID_DETECT_NUM * cfg.PYRAMID_DETECT_INDEX
                    endIndex = startIndex + cfg.PYRAMID_DETECT_NUM - 1
                    scores = scores[startIndex:endIndex]
                    boxes = boxes[startIndex:endIndex]
                    keyPoints = keyPoints[startIndex:endIndex]

            boxesList.append(boxes.copy())
            scoresList.append(scores.copy())
            keyPointsList.append(keyPoints.copy())
            agesList.append(age_pros.copy())
            gendersList.append(gender_pros.copy())
            ethricitiesList.append(etnricity_pros.copy())

        for cls_ind, cls in enumerate(CLASSES[1:]):
            detsList = []
            det_keyPointsList = []
            det_agesList = []
            det_gendersList = []
            det_ethricitiesList = []
            if modelType != 'rpn':
                cls_ind += 1  # because we skipped background
            for boxes, scores, keyPoints, ages, genders, ethricities in zip(boxesList, scoresList, keyPointsList, agesList, gendersList, ethricitiesList):
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                keyPoints_num = keyPoints.shape[1] / 2
                cls_keyPoints = keyPoints[:, keyPoints_num * cls_ind:keyPoints_num * (cls_ind + 1)]
                cls_ages = np.argmax(ages, 1)
                cls_genders = np.argmax(genders, 1)
                cls_ethricities = np.argmax(ethricities, 1)
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                det_keyPoints = cls_keyPoints[keep, :]
                det_ages = cls_ages[keep]
                det_genders = cls_genders[keep]
                det_ethricities = cls_ethricities[keep]

                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                if detsList == []:
                    detsList = dets[inds, :].copy()
                    det_keyPointsList = det_keyPoints[inds, :].copy()
                    det_agesList = det_ages[inds].copy()
                    det_gendersList = det_genders[inds].copy()
                    det_ethricitiesList = det_ethricities[inds].copy()
                else:
                    detsList = np.vstack((detsList, dets[inds, :]))
                    det_keyPointsList = np.vstack((det_keyPointsList, det_keyPoints[inds, :]))
                    det_agesList = np.vstack((det_agesList, det_ages[inds]))
                    det_gendersList = np.vstack((det_gendersList, det_genders[inds]))
                    det_ethricitiesList = np.vstack((det_ethricitiesList, det_ethricities[inds]))

            # dets with different scale do nms
            keep = nms(detsList, NMS_THRESH)
            dets = detsList[keep, :]
            det_keyPoints = det_keyPointsList[keep, :]
            det_ages = det_agesList[keep]
            det_genders = det_gendersList[keep]
            det_ethricities = det_ethricitiesList[keep]

            # record result
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                keyPoints = det_keyPoints[i, :keyPoints_num]
                keyPoints_str = ['%.1f' % key for key in keyPoints]
                age = det_ages[i]
                gender = det_genders[i]
                ethricity = det_ethricities[i]
                resultList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:s} {:d} {:d} {:d}\n'.
                                 format(im_path, score,
                                        bbox[0] + 1, bbox[1] + 1,
                                        bbox[2] + 1, bbox[3] + 1,
                                        ' '.join(keyPoints_str),
                                        age, gender, ethricity))
    resultList.close()


# region generate FaceDB about afw, helen, lfpw etc.

def generateFaceDB(DBName, prototxt, modelPath, modelType, CONF_THRESH=0.8,
                   NMS_THRESH=0.3, scales=[0], saveDIR=None, DBtype='trainval',
                   recompute=1, visual=1):
    # load initial parameter
    gpu_id = 0
    cfg.TEST.HAS_RPN = True
    cfg.GPU_ID = gpu_id
    # initialize net
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    imgsInfo = _loadFaceDB(DBName, DBtype)

    saveFaceDBDIR = os.path.join(saveDIR, DBName, DBtype)
    mk_dir(saveFaceDBDIR, 0)
    DetectFilePath = os.path.join(saveFaceDBDIR, 'detectInfo.txt')
    detImgsInfo = _transformFaceDB(imgsInfo, prototxt, modelPath, modelType, DetectFilePath,
                     CONF_THRESH, NMS_THRESH, scales, recompute)

    if visual:
        detImageSaveDir = os.path.join(saveFaceDBDIR, 'detectDir')
        visual_result_300w_face(imgsInfo, detImgsInfo, detImageSaveDir)

    # saveFaceDB
    _saveFaceDB(imgsInfo, saveFaceDBDIR)


def _loadFaceDB(DBName, DBtype='trainval'):
    print 'load %s FaceDB with %s ..' % (DBName, DBtype)
    originalDBDir = os.path.join(threeHusOtherDBDir, DBName)
    for dir, subdir, files in os.walk(originalDBDir):
        if len(subdir) != 0:
            assert 'testset' in subdir and 'trainset' in subdir, 'testset and trainset are required'
            print 'load %s...' % DBtype
            if DBtype == 'trainset' or DBtype == 'testset':
                DBfiles = os.listdir(os.path.join(originalDBDir, DBtype))
                DBfiles = [DBtype+'/'+t for t in DBfiles]
            elif DBtype == 'trainval':
                trainsetFiles = os.listdir(os.path.join(originalDBDir, 'trainset'))
                trainsetFiles = ['trainset'+'/'+t for t in trainsetFiles]
                testsetFiles = os.listdir(os.path.join(originalDBDir, 'testset'))
                testsetFiles = ['testset'+'/'+t for t in testsetFiles]
                DBfiles = trainsetFiles + testsetFiles
            else:
                raise customError('%s is invalid' % DBtype)
        else:
            assert DBtype == 'trainval', 'the DB only support trainval'
            print 'load trainval..'
            DBfiles = files

        # load imgsInfo
        suffixList = list(set([DBfile.split('.')[1] for DBfile in DBfiles]))
        imgSuffix = filter(lambda k: k in ['png', 'jpg'], suffixList)[0]
        label_files = filter(lambda f: f.endswith('.pts'), DBfiles)
        img_files = filter(lambda f: f.endswith('.'+imgSuffix), DBfiles)

        if DBName == 'afw':
            unilabel_files = list(set([file.split('_')[0] for file in label_files]))
        elif DBName == 'helen' or DBName == 'lfpw':
            # one image one label
            unilabel_files = [file.split('.')[0] for file in label_files]
        elif DBName == 'ibug':
            unilabel_files = [file.split('.')[0] for file in filter(lambda f:len(f.split('_'))==2, label_files)]
        imgsInfo = {}
        for unilabel_file in unilabel_files:
            imgPath = filter(lambda f: f.startswith(unilabel_file), img_files)[0]
            imgPath = os.path.join(dir, imgPath)
            assert os.path.exists(imgPath)
            labelNames = filter(lambda f: f.startswith(unilabel_file), label_files)
            labels = np.zeros([len(labelNames), 68, 2])
            for labelIndex, labelName in enumerate(labelNames):
                labelPath = os.path.join(dir, labelName)
                assert os.path.exists(labelPath)
                results = linecache.getlines(labelPath)
                results = np.array(([x[:-1].split(' ')[:2] for x in results[3:71]]), dtype=np.float)
                labels[labelIndex, :, :] = results
            imgsInfo[imgPath] = labels

        return imgsInfo


def _transformFaceDB(imgsInfo, prototxt, modelPath, modelType, DetectFilePath,
                     CONF_THRESH, NMS_THRESH, scales, recompute):
    print 'detect FaceDB using existed model..'
    if not os.path.exists(DetectFilePath) or recompute:
        det_model_facePlus_pyramid(prototxt, modelPath, modelType, imgsInfo.keys(),
                                   DetectFilePath, CONF_THRESH, NMS_THRESH, scales=scales)
    print 'load detected FaceDB..'
    detImgsInfo = loadDetectFile_facePlus(DetectFilePath, imgsInfo.keys())

    print 'transform new FaceDB..'
    for imgPath, detFaces in detImgsInfo.items():
        if len(detFaces) == 0:
            imgsInfo.pop(imgPath)
            continue
        newInfo = np.empty(detFaces.shape, np.float)
        newInfo.fill(-1)
        for i, detFace in enumerate(detFaces):
            for imgInfo in imgsInfo[imgPath]:
                x_valid_num = len(np.where((imgInfo[17:, 0] > detFace[0]) &
                                           (imgInfo[17:, 0] < detFace[2]))[0])
                y_valid_num = len(np.where((imgInfo[17:, 1] > detFace[1]) &
                                           (imgInfo[17:, 1] < detFace[3]))[0])
                avg_valid_num = (x_valid_num + y_valid_num) / 2
                if avg_valid_num > 46:
                    newInfo[i, 5:] = imgInfo.ravel()
            newInfo[i, 0:5] = detFace[0:5]

        imgsInfo[imgPath] = newInfo

    return detImgsInfo


def _saveFaceDB(imgsInfo, saveFaceDBDIR):
    print 'save transformed FaceDB..'
    # save results
    imgSaveDIR = os.path.join(saveFaceDBDIR, 'imdb')
    attributeSaveDIR = os.path.join(saveFaceDBDIR, 'imdb_attribute')
    mk_dir(imgSaveDIR, 1)
    mk_dir(attributeSaveDIR, 1)
    for imgPath, faces in imgsInfo.items():
        imgName = imgPath.split('/')[-1]
        imgInfoName = imgName.split('.')[0] + '.txt'
        imgInfoPath = os.path.join(attributeSaveDIR, imgInfoName)
        newImgPath = os.path.join(imgSaveDIR, imgName)
        with open(imgInfoPath, 'w') as result:
            for face in faces:
                result.write(' '.join(face.astype(np.str)) + '\n')
        shutil.copyfile(imgPath, newImgPath)

# endregion


def showImg(im, dets):
    for i in range(len(dets)):
        x1 = dets[i][0]
        y1 = dets[i][1]
        x2 = dets[i][2]
        y2 = dets[i][3]
        score = dets[i][-1]
        scale = 0.8  # (10/(bbox[2]-bbox[0]))
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # cv2.putText(im, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
        # cv2.putText(im, str(score), (x1, int(y2+15)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
    # cv2.imshow("Image", im)
    # cv2.waitKey(0)
    return im


if __name__ == '__main__':
    aps = []
    for strategyThresh in np.arange(-0.20, -0.21, -0.01):
        for thresh in np.arange(20000, 30000, 10000):  # 0.1, 1, 0.1 100000, 0, -10000
            for scale in [2.5]:  # np.arange(2.5, 2.6, 0.1):
                # region parameter initialization
                TrainingDB = 'threeHusFace'  # wider_face moon face_plus threeHusFace
                save_dir = 'output/test_result/%s' % TrainingDB
                save_dir = os.path.join(cfg.ROOT_DIR, save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                scales = [scale]  # 0, 200, 600 1, 0.5, 2
                # cfg.TEST.RPN_POST_NMS_TOP_N = 150
                # custom parameters
                # cfg.TEST.SCALES = (550,)  # 600 800 1000 1200 1400
                # cfg.TEST.MAX_SIZE = 950  # 1000 1200 1400 1600 1800
                suffix = '_m_%.1f' % scale  # _Annotations_new_v2 _a_0_200_600 _p4_3
                # _s300_500 _s3 _s4 _s7_s3_a _s13_s4 _s11_s3 _a1
                ''' wider face '''
                # method = 'VGG16_faster_rcnn_stage4_iter_40000_v3'
                # method = 'VGG16_rpn_stage4_iter_80000_v3'
                # method = 'VGG16_rpn_v1_stage1_iter_80000_v3'
                # method = 'VGG16_rpn_stage1_iter_80000_v3'
                # method = 'VGG16_faster_rcnn_end2end_iter_70000_v2'
                # method = 'ResNet-50_rfcn_end2end_ohem_iter_70000_v2'
                # method = 'VGG16_faster_rcnn_end2end_with_conv4_iter_70000_v4'
                # method = 'VGG16_faster_rcnn_end2end_with_fuse_iter_90000_v3'  # 90000_v7
                # method = 'VGG16_faster_rcnn_end2end_with_multianchor_iter_90000_v3_6_voc'
                # method = 'VGG16_faster_rcnn_end2end_with_fuse_multianchor_iter_110000_v2_2_voc'  # v1_2 v2 v2_1_voc v2_2_voc
                # method = 'VGG16_faster_rcnn_end2end_with_multianchor_iter_70000_v3_1_voc'
                # method = 'VGG16_faster_rcnn_end2end_with_pyramid_2_iter_110000_v2_1_voc'
                # method = 'VGG16_faster_rcnn_end2end_with_multianchor_iter_100000_v3-3_fc_voc_nf'
                # method = 'VGG16_faster_rcnn_end2end_with_multianchor_iter_100000_v3-3_fc_fp1'
                # method = 'VGG16_faster_rcnn_end2end_with_multianchor_v3-3_fc_fp1_2_iter_100000'
                # method = 'VGG16_faster_rcnn_end2end_with_multianchor_v3_1_voc_fp1_iter_90000'
                # method = 'VGG16_faster_rcnn_end2end_with_multianchor_v3-3_lha_iter_100000'
                # method = 'VGG16_faster_rcnn_end2end_with_fuse_multianchor_v2_lha_wfp1_iter_50000'
                # method = 'VGG16_faster_rcnn_end2end_with_fuse_multianchor_v2_2_voc_fp1_2_iter_100000'
                # method = 'VGG16_faster_rcnn_end2end_with_fuse_multianchor_v2_lha_wfp1_2_s2_iter_90000'

                method = 'VGG16_faster_rcnn_end2end_with_fuse_multianchor_v2-4-1_voc_fp1_2_iter_100000'
                ''' moon '''
                # method = 'VGG16_faster_rcnn_end2end_with_multianchor_iter_20000_v3'

                modelDir = 'model'
                testDataSet = 'wider'  # wider FDDB Face_Plus threeHusFace Face_Plus_v1
                # testDataSet = 'moon'
                # modelDir = 'faster_rcnn_alt_opt/v3/s11'  # s2 s4 s5 s7 s8 s13
                methodType = 'normal'  # rpn normal
                ''' wider face '''
                # prototxt = 'VGG16/faster_rcnn_end2end/test.prototxt'
                # prototxt = 'VGG16/faster_rcnn_end2end/test_fuse_v3.prototxt'
                # prototxt = 'ResNet-50/rfcn_end2end_ohem/test_agnostic.prototxt'
                # prototxt = 'VGG16/faster_rcnn_alt_opt/rpn_test.pt'
                # prototxt = 'VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
                # prototxt = 'VGG16/faster_rcnn_end2end/test_multianchor_v3.prototxt'
                # prototxt = 'VGG16/faster_rcnn_end2end/test_fuse_multianchor_v2.prototxt'
                # prototxt = 'VGG16/faster_rcnn_end2end/test_fuse_multianchor_v2-2.prototxt'
                # prototxt = 'VGG16/faster_rcnn_end2end/test_pyramid_v2.prototxt'VGG16_faster_rcnn_end2end_with_multianchor_iter_100000_v3-3-2-1_fc_voc.caffemodel
                # prototxt = 'VGG16/faster_rcnn_end2end/68_keyPoints/test_multianchor_v3-3_t1.prototxt'
                # prototxt = 'VGG16/faster_rcnn_end2end/68_keyPoints/test_fuse_multianchor_v2_t1.prototxt'
                # prototxt = 'VGG16/faster_rcnn_end2end/68kp+age+race+sex/test_multianchor_v3-6_t2.prototxt'
                # prototxt = 'VGG16/faster_rcnn_end2end/68_keyPoints/test_fuse_multianchor_v2_t1_s2.prototxt'
                prototxt = 'VGG16/faster_rcnn_end2end/test_fuse_multianchor_v2-4-1.prototxt'

                ''' moon '''
                # prototxt = 'VGG16/faster_rcnn_end2end/test_multianchor_v3.prototxt'

                # adjust height of face
                adjustBox = False  # True
                adjustRate = 0.1

                if adjustBox:
                    adjustBoxStr = '_ad%03d' % (adjustRate * 100)
                else:
                    adjustBoxStr = ''

                Adapter = False  # True False
                if Adapter:
                    IM_GT_NUM = 0
                    CONF_THRESH = 0.5
                    NMS_THRESH = 0.3
                else:
                    CONF_THRESH = 0.1  # 0.8:2 0.9:4
                    NMS_THRESH = 0.3  # 0.3
                # endregion

                # region segment setting
                SegStrategy = False  # False True
                if SegStrategy:
                    strategyType = 's3'  # s1
                    filterThresh = 0.7  # 0.7
                    outClassNum = 2  # 21
                    segMethod = 'fcn32s_%d_%02d%s' % (thresh, filterThresh * 10, adjustBoxStr)
                    segWeight = strategyThresh  # strategyThresh 0.1
                    overSegThresh = 0.5
                    methodName = method + '_c%d_n%d%s' % (CONF_THRESH * 10, NMS_THRESH * 10, suffix)
                    if segWeight < 0:
                        segMethodStr = '_%s_w%04d_%s_cn%d' % (segMethod, segWeight * 100, strategyType, outClassNum)
                    else:
                        segMethodStr = '_%s_w%03d_%s_cn%d' % (segMethod, segWeight * 100, strategyType, outClassNum)
                    fuseDetectFilePath = save_dir + '/FuseDetect/' + methodName + segMethodStr + '.txt'
                    segPrototxt = '/data4/yyliang/face/fcn.berkeleyvision.org/wider-fcn32s/deploy.prototxt'
                    segModelPath = '/data4/yyliang/face/fcn.berkeleyvision.org/wider-fcn32s/snapshot/score_2/train_iter_%d.caffemodel' % thresh
                else:
                    strategyType = ''
                    segMethod = ''
                    segMethodStr = ''
                    segWeight = 0.1
                    filterThresh = 0.0
                    segPrototxt = None
                    segModelPath = None
                # endregion

                # region filePath auto match
                methodName = method + '_c%d_n%d%s' % (CONF_THRESH * 10, NMS_THRESH * 10, suffix)
                modelPath = '%s/%s.caffemodel' % (modelDir, method)
                modelPath = os.path.join(cfg.ROOT_DIR, 'output/%s' % TrainingDB, modelPath)
                prototxt = os.path.join(cfg.ROOT_DIR, 'models/%s' % TrainingDB, prototxt)
                detectSaveDir = save_dir + '/Detect/'
                detectFilePath = detectSaveDir + methodName + '.txt'
                widerValDetectSaveDir = save_dir + '/Voc_val_Detect/'
                widerValDetectFilePath = widerValDetectSaveDir + methodName + '.txt'
                widerDetectSaveDir = save_dir + '/Wider_val_Detect/' + methodName

                facePlusValDetectSaveDir = save_dir + '/FPs_val_Detect/'
                facePlusValDetectFilePath = facePlusValDetectSaveDir + methodName + '.txt'

                threeHusValDetectSaveDir = save_dir + '/THs_val_Detect/'
                threeHusValDetectFilePath = threeHusValDetectSaveDir + methodName + '.txt'

                metricSaveDir = save_dir + '/Metric/' + testDataSet
                metricFilePath = metricSaveDir + '/Metric_' + methodName + segMethodStr + '.txt'

                # metricFilePath = metricSaveDir + '/Metric_FRCNN_MINE' + '' + segMethodStr + '.txt'

                tpfpSaveDir = save_dir + '/TPFP/' + testDataSet
                tpfpFilePath = tpfpSaveDir + '/TPFP_' + methodName + segMethodStr + '.jpg'
                imfpSaveDir = save_dir + '/IMFP'
                imfpFilePath = imfpSaveDir + '/IMFP_' + methodName + segMethodStr + '.txt'
                sortedImfpSaveDir = save_dir + '/SortedIMFP'
                sortedImfpFilePath = sortedImfpSaveDir + '/SortedIMFP_' + methodName + segMethodStr + '.txt'
                sortedImtpSaveDir = save_dir + '/SortedIMTP'
                sortedImtpFilePath = sortedImtpSaveDir + '/SortedIMTP_' + methodName + segMethodStr + '.txt'
                # widerImageSaveDir = save_dir + '/Wider_IMG/IMG_' + methodName + segMethodStr
                detImageSaveDir = save_dir + '/Det_IMG/' + testDataSet + '/IMG_' + methodName + segMethodStr
                imageSaveDir = save_dir + '/IMG/IMG_' + methodName + segMethodStr

                mk_dir(detectSaveDir, 0)
                mk_dir(widerValDetectSaveDir, 0)
                mk_dir(facePlusValDetectSaveDir, 0)
                mk_dir(threeHusValDetectSaveDir, 0)
                mk_dir(metricSaveDir, 0)
                # mk_dir(detImageSaveDir, 1)
                mk_dir(tpfpSaveDir, 0)
                mk_dir(imfpSaveDir, 0)
                mk_dir(sortedImfpSaveDir, 0)
                mk_dir(sortedImtpSaveDir, 0)

                # endregion

                # main run
                timer = Timer()
                timer.tic()

                metricList = generate_result_v1(modelPath, prototxt, testDataSet, methodType, recompute=1,
                                                CONF_THRESH=CONF_THRESH, NMS_THRESH=NMS_THRESH,
                                                visual=0, recordIMFP=0, Adapter=Adapter,
                                                segModelPath=segModelPath, segPrototxt=segPrototxt,
                                                segWeight=segWeight, strategyType=strategyType,
                                                scales=scales, metricFilePath=metricFilePath,
                                                tpfpFilePath=tpfpFilePath, includeRPN=1)
                print metricList[-1]

                timer.toc()
                print ('total took {:.3f}s').format(timer.total_time)
                print 'done'



                # region Single Case (including test case)

                # 1.test generate_result_voc
                # vocValImgList = cfg.ROOT_DIR + '/data/MOONdevkit/data/ImageSets/Main/train.txt'
                # vocValAnnoPath = cfg.ROOT_DIR + '/data/MOONdevkit/data/Annotations/{:s}.xml'
                # vocValImgDir = cfg.ROOT_DIR + "/data/MOONdevkit/data/JPEGImages/"
                # vocValDetectFilePath = widerValDetectFilePath
                # metricList = generate_result_voc(modelPath, prototxt, vocValAnnoPath, vocValImgDir,
                #                                  vocValImgList, vocValDetectFilePath, methodType,
                #                                  recompute=1, CONF_THRESH=CONF_THRESH, NMS_THRESH=NMS_THRESH,
                #                                  visual=1, recordIMFP=0)
                # record_result(metricList, metricFilePath, tpfpFilePath)

                # 2.test generate_result_gen_val
                # cfg.TRAIN.ValImgList = threeHusTestDir
                # cfg.TRAIN.ValImgList = cfg.ROOT_DIR + '/data/DB/object/voc_wider/ImageSets/Main/val.txt'
                # cfg.TRAIN.ValAnnoPath = cfg.ROOT_DIR + '/data/DB/object/voc_wider/Annotations/{:s}.xml'
                # cfg.TRAIN.vocValImgDir = cfg.ROOT_DIR + "/data/DB/object/voc_wider/JPEGImages/"
                # cfg.TRAIN.METRIC_RPN = True
                # metricList = generate_result_gen_val(modelPath, prototxt, 'test', testDataSet, cfg.TRAIN.ValImgList, recompute=1,
                #                                 CONF_THRESH=CONF_THRESH, NMS_THRESH=NMS_THRESH,
                #                                 visual=0, recordIMFP=0)
                # print metricList

                # 3.generate FaceDB
                # saveDIR = '/home/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/300-w_face/otherDB/save/'
                # generateFaceDB('ibug', prototxt, modelPath, methodType, CONF_THRESH=0.97,
                #                NMS_THRESH=NMS_THRESH, DBtype='trainval', recompute=0,
                #                saveDIR=saveDIR, visual=1)
                # print 'done'

                # 4.test generate_result
                # metricList = generate_result(modelPath, prototxt, methodType, recompute=0,
                #                              CONF_THRESH=CONF_THRESH, NMS_THRESH=NMS_THRESH,
                #                              visual=1, recordIMFP=1, Adapter=Adapter)
                # print 'modelPath:%s\nprototxt:%s\nmethodType:%s' % (modelPath, prototxt, methodType)
                # ap = generate_result(modelPath, prototxt, methodType, recompute=1)[0]
                # print 'ap %.8f\n' % ap # 0.91791757

                # other test
                # matFile = '/home/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/wider/wider_face_val-v7.mat'
                # save_dir = os.path.join(cfg.ROOT_DIR, 'output/test_result')
                # loadWidervalAndDetect()
                # loadROCAndPlot('/home/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/test_result/RocFiles')
                # gather_result()

                # endregion

