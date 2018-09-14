# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by sky
# --------------------------------------------------------
import _init_ms_paths
import os
import cv2
import matplotlib.pyplot as plt
from MultiAttributeDB.file_tool import *
from utils.MultiAttributeDB.FaceImage import FaceImage

def loadFaceAttribute(AttriFile):
    with open(AttriFile, 'r') as f:
        lines = f.readlines()
    attriInfo = [line.split(' ') for line in lines]
    return attriInfo

def transformFileFormat(ImdbFiles, ImdbType='imdb'):
    if ImdbType == 'imdb':
        ImgDir = '/data5/dataset/imdb_attributes/imdb/'
        AttriDir = '/data5/dataset/imdb_attributes/imdb_attributes/'
    elif ImdbType == 'imdb2':
        ImgDir = '/data5/dataset/imdb_attributes/imdb2/'
        AttriDir = '/data5/dataset/imdb_attributes/imdb2_attributes/'
    else:
        print 'type is invalid'
        exit(1)
    ImgFiles = []
    AttriFiles = []
    for ImdbFile in ImdbFiles:
        ImdbFileInfo = ImdbFile.split('/')
        ImgFile = ImgDir + '/'.join(ImdbFileInfo[3:])
        AttriFile = AttriDir + '/'.join(ImdbFileInfo[3:-1]) + '/' + ImdbFileInfo[-1].split('.')[0] + '.txt'
        assert os.path.exists(ImgFile), 'Path does not exist: {}'.format(ImgFile)
        assert os.path.exists(AttriFile), 'Path does not exist: {}'.format(AttriFile)
        ImgFiles.append(ImgFile)
        AttriFiles.append(AttriFile)
    return ImgFiles, AttriFiles

def visualFaceAttribute(trainImdbImgFiles, trainImdbAttriFiles):
    # initialization setting
    sum = 0
    outlier_error_sum = 0
    attriInfos_none_sum = 0
    attriInfo_none_sum = 0
    label_error_sum = 0
    attriInfos_none_logs = []
    for trainImdbImgFile, trainImdbAttriFile in zip(trainImdbImgFiles, trainImdbAttriFiles):
        sum = sum + 1
        # plt.figure(1)
        img = cv2.imread(trainImdbImgFile)
        height, width = img.shape[:-1]
        attriInfos = loadFaceAttribute(trainImdbAttriFile)

        if attriInfos == [['None\n']]:
            attriInfos_none_sum = attriInfos_none_sum + 1
            attriInfos_none_logs.append(trainImdbImgFile)
            continue
        for attriInfo in attriInfos:
            if attriInfo == ['None\n']:
                attriInfo_none_sum = attriInfo_none_sum + 1
                continue
            x1 = int(attriInfo[26])
            y1 = int(attriInfo[27])
            x2 = x1 + int(attriInfo[28])
            y2 = y1 + int(attriInfo[25])

            # outlier error
            if x1 < 0 or y1 < 0 or x2 > width-1 or y2 > height-1:
                outlier_error_sum = outlier_error_sum + 1
                print 'ImgPath: %s\nAttriPath: %s' % (trainImdbImgFile, trainImdbAttriFile)
                print 'x1,y1,x2,y2 = (%d, %d, %d, %d)' % (x1, y1, x2, y2)
                print 'height, width = (%d, %d)' % (height, width)
                # modify outers
                x1 = min(max(0, x1), width-1)
                y1 = min(max(0, y1), height-1)
                x2 = min(max(0, x2), width-1)
                y2 = min(max(0, y2), height-1)

            # label error
            if x1 == x2 or y1 == y2:
                label_error_sum = label_error_sum + 1
                print 'ImgPath: %s\nAttriPath: %s' % (trainImdbImgFile, trainImdbAttriFile)
                print 'x1,y1,x2,y2 = (%d, %d, %d, %d)' % (x1, y1, x2, y2)
                print 'height, width = (%d, %d)' % (height, width)
                # exit(0)
                continue

        #     scale = 1  # (10/(bbox[2]-bbox[0]))
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #     cv2.putText(img, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 1)

        # ax = plt.subplot(1, 1, 1)
        # ax.imshow(img[:, :, ::-1], aspect='equal')
        print 'sum: %d' % sum
        print 'outlier_error_sum: %d' % outlier_error_sum
        print 'attriInfos_none_sum: %d' % attriInfos_none_sum
        print 'attriInfo_none_sum: %d' % attriInfo_none_sum
        print 'label_error_sum: %d' % label_error_sum
    # return logs
    assert len(attriInfos_none_logs) == attriInfos_none_sum, 'not match for attriInfos_none_logs and attriInfos_none_sum'
    return attriInfos_none_logs

def checkFaceAttribute(trainImdbImgFiles, trainImdbAttriFiles, saveFilePath):
    # initialization setting
    sum = 0
    attriInfos_none_sum = 0
    attriInfos_none_logs = []
    ages_none_sum = 0
    ages_invalid_sum = 0
    gender_none_sum = 0
    gender_invalid_sum = 0
    ethnicity_none_sum = 0
    ethnicity_invalid_sum = 0
    point_none_sum = 0

    for trainImdbImgFile, trainImdbAttriFile in zip(trainImdbImgFiles, trainImdbAttriFiles):
        sum = sum + 1
        attriInfos = loadFaceAttribute(trainImdbAttriFile)

        # check if all attriInfos is None
        if attriInfos == [['None\n']]:
            attriInfos_none_sum = attriInfos_none_sum + 1
            attriInfos_none_logs.append(trainImdbImgFile)
            continue
        # check if single attriInfo is None or unreasonable
        for attriInfo in attriInfos:
            if attriInfo[0] == 'None':
                ages_none_sum = ages_none_sum + 1
            else:
                if float(attriInfo[0]) < 0 or float(attriInfo[1]) > 100:
                    ages_invalid_sum = ages_invalid_sum + 1
            if attriInfo[3] == 'None':
                ethnicity_none_sum = ethnicity_none_sum + 1
            else:
                if attriInfo[3] not in ['White', 'Asian', 'Black']:
                    ethnicity_invalid_sum = ethnicity_invalid_sum + 1
            if attriInfo[18] == 'None':
                gender_none_sum = gender_none_sum + 1
            else:
                if attriInfo[18] not in ['Female', 'Male']:
                    gender_invalid_sum = gender_invalid_sum + 1
            for point, index in enumerate(attriInfo[30:196]):
                # if index % 2 == 0 and point < 0:
                #     point_invalid_sum = point_invalid_sum + 1
                # if index % 2 == 1 and point > 0:
                if point == 'None':
                    point_none_sum = point_none_sum + 1


        print 'sum: %d' % sum
        print 'attriInfos_none_sum: %d' % attriInfos_none_sum
        print 'ages_none_sum: %d' % ages_none_sum
        print 'ages_invalid_sum: %d' % ages_invalid_sum
        print 'ethnicity_none_sum: %d' % ethnicity_none_sum
        print 'ethnicity_invalid_sum: %d' % ethnicity_invalid_sum
        print 'gender_none_sum: %d' % gender_none_sum
        print 'gender_invalid_sum: %d' % gender_invalid_sum
        print 'point_none_sum: %d' % point_none_sum

    # return logs
    print 'sum: %d' % sum
    print 'attriInfos_none_sum: %d' % attriInfos_none_sum
    print 'ages_none_sum: %d' % ages_none_sum
    print 'ages_invalid_sum: %d' % ages_invalid_sum
    print 'ethnicity_none_sum: %d' % ethnicity_none_sum
    print 'ethnicity_invalid_sum: %d' % ethnicity_invalid_sum
    print 'gender_none_sum: %d' % gender_none_sum
    print 'gender_invalid_sum: %d' % gender_invalid_sum
    print 'point_none_sum: %d' % point_none_sum
    assert len(attriInfos_none_logs) == attriInfos_none_sum, 'not match for attriInfos_none_logs and attriInfos_none_sum'
    recordLogs(saveFilePath, attriInfos_none_logs)

def getValidSamples(trainImdbImgFiles, trainImdbAttriFiles):
    # initialization setting
    sum = 0
    attriInfos_none_sum = 0
    validSamples = []

    for trainImdbImgFile, trainImdbAttriFile in zip(trainImdbImgFiles, trainImdbAttriFiles):
        sum = sum + 1
        # attriInfos = loadFaceAttribute(trainImdbAttriFile)
        attriInfos = FaceImage(trainImdbAttriFile)

        # check if all attriInfos is None
        if attriInfos == [['None\n']]:
            attriInfos_none_sum = attriInfos_none_sum + 1
            continue
        else:
            validSamples.append(trainImdbImgFile)
        print 'sum: %d' % sum
    # return logs
    print 'sum: %d' % sum
    print 'attriInfos_none_sum: %d' % attriInfos_none_sum
    print 'validSamples_sum: %d' % len(validSamples)
    return validSamples

def run_check(ImdbFile, ImdbType, saveFilePath):
    with open(ImdbFile) as f:
        ImdbFiles = [x.strip() for x in f.readlines()]

    ImdbImgFiles, ImdbAttriFiles = transformFileFormat(ImdbFiles, ImdbType)

    checkFaceAttribute(ImdbImgFiles, ImdbAttriFiles, saveFilePath)

def run_visual(ImdbFile, ImdbType):
    with open(ImdbFile) as f:
        ImdbFiles = [x.strip() for x in f.readlines()]

    ImdbImgFiles, ImdbAttriFiles = transformFileFormat(ImdbFiles[:100], ImdbType)

    visualFaceAttribute(ImdbImgFiles, ImdbAttriFiles)

def recordLogs(saveFilePath, logs):
    with open(saveFilePath, 'w') as f:
        for log in logs:
            f.write(log + '\n')

def run_filter(ImdbFile, ImdbType, saveFilePath, NoneFilter = 1):
    with open(ImdbFile) as f:
        ImdbFiles = [x.strip() for x in f.readlines()]

    ImdbImgFiles, ImdbAttriFiles = transformFileFormat(ImdbFiles, ImdbType)
    if NoneFilter:
        validSamples = getValidSamples(ImdbImgFiles[:80000], ImdbAttriFiles[:80000])
        recordLogs(saveFilePath, validSamples)
        print 'num of record is %d' % len(validSamples)
    else:
        recordLogs(saveFilePath, ImdbImgFiles)
        print 'num of record is %d' % len(ImdbImgFiles)

def run_washed_filter(ImdbTypeList, saveFilePath, NoneFilter = 1):
    # assert ImdbType in ['imdb', 'imdb2', 'all']
    ImdbImgFiles = get_latest_img_list(ImdbTypeList)
    ImdbAttriFiles = get_latest_lbl_list(ImdbTypeList)
    if NoneFilter:
        validSamples = getValidSamples(ImdbImgFiles, ImdbAttriFiles)
        recordLogs(saveFilePath, validSamples)
        print 'num of record is %d' % len(validSamples)
    else:
        recordLogs(saveFilePath, ImdbImgFiles)
        print 'num of record is %d' % len(ImdbImgFiles)

if __name__ == '__main__':
    # trainImdbFile = "/data5/dataset/imdb_attributes/imdb.txt"
    # testImdbFile = "/data5/dataset/imdb_attributes/imdb2.txt"
    # saveFilePath = '/home/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/Face_plus/train_unwash.txt'
    # run_filter(testImdbFile, 'imdb2', saveFilePath, 1)

    # washed imdb
    # facePlus(imdb, imdb2)
    # saveFilePath = '/home/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/Face_plus/test.txt'
    # run_washed_filter('imdb2', saveFilePath, NoneFilter=1)
    # saveFilePath = '/home/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/Face_plus/test.txt'
    # saveFilePath = '/data6/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/Face_plus/train.txt'
    # run_washed_filter('facepp_train', saveFilePath, NoneFilter=1)
    # run_washed_filter(['facepp_test', 'facepp_val'], saveFilePath, NoneFilter=1)

    # 300-w face
    saveFilePath = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/COFW/train/train.txt'
    # saveFilePath = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/AFLW/debug.txt'
    # saveFilePath = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/train.txt'
    # 'afw_trainval', 'lfpw_trainset', 'helen_trainset'
    # 'aflw-full_trainset' '300w-c(re)_trainval' 'lfpw_testset', 'afw_trainval', 'helen_testset', 'lfpw_trainset', 'helen_trainset', 'ibug_trainval'
    # '300w(re)_trainval'
    run_washed_filter(['cofw_trainset'], saveFilePath, NoneFilter=1)

    # 300-w face
    # # saveFilePath = '/data5/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/300-w_face/train_test.txt'
    # saveFilePath = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/train_test.txt'
    # # 'afw_trainval', 'lfpw_trainset', 'helen_trainset'
    # # 'lfpw_testset', 'afw_trainval', 'helen_testset', 'lfpw_trainset', 'helen_trainset', 'ibug_trainval'
    # run_washed_filter(['lfpw_testset', 'afw_trainval', 'helen_testset', 'lfpw_trainset', 'helen_trainset', 'ibug_trainval', 'frgc_trainval'], saveFilePath, NoneFilter=1)


    # heterology data
    # saveFilePath = '/data5/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/Face_plus/train.txt'
    # # 'afw_trainval', 'lfpw_trainset', 'helen_trainset'
    # run_washed_filter(['wider_train', 'morph_s1_train', 'morph_s2_train', 'afw_trainval', 'lfpw_trainset', 'helen_trainset', 'ibug_trainval'], saveFilePath, NoneFilter=1)

    # morph
    # saveFilePath = '/data6/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/DB/face/Morph/S2/test.txt'
    # run_washed_filter(['morph_s2_test'], saveFilePath, NoneFilter=1)
