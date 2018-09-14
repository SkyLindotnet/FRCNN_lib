# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by sky
# --------------------------------------------------------
import numpy as np
import random
import os
import scipy.stats as st
import matplotlib.pyplot as plt
from timer import Timer

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def asign_train_test(num, rate=1, format=6):
    str = '%%0%dd' % format
    sampleList = range(num)
    samples = range(1, num+1)
    sampleStrList = np.array([str % i for i in samples])
    trainNum = int(num * rate)
    trainList = random.sample(sampleList, trainNum)
    testList = list(set(sampleList) ^ set(trainList))
    trainStrList = sampleStrList[trainList]
    testStrList = sampleStrList[testList]
    print 'trainStrList'
    print trainStrList
    print 'testStrList'
    print testStrList
    return trainStrList, testStrList

def rename_files(dir, format=6):
    for dir_name, sub_dirs, file_list in os.walk(dir):
        for file in file_list:
            newName = '%06d' % int(file.split('.')[0])
            os.rename(os.path.join(dir_name, file), os.path.join(dir_name, newName+'.'+file.split('.')[1]))

def gkern(kernlen=33, nsig=5):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def getHeatmapBygkern(orgmaps, kernlen=33, nsig=5, straget=1):
    # timer = Timer()
    # timer.tic()
    heatmaps = np.zeros([orgmaps.shape[0], orgmaps.shape[1], orgmaps.shape[2]])
    kernmap = gkern(kernlen, nsig)
    for i, orgmap in enumerate(orgmaps):
        assert len(orgmap.shape) == 2
        assert (kernlen-1) % 2 == 0
        addlen = (kernlen-1) / 2
        height, width = orgmap.shape
        newmap = np.zeros([height+2*addlen, width+2*addlen])
        newmap[addlen:addlen+height, addlen:addlen+width] = orgmap
        kp_i = np.where(newmap == 1)
        assert len(kp_i[0]) < 2
        if len(kp_i[0]) == 1:
            kp_x = kp_i[0][0]-addlen
            kp_y = kp_i[1][0]-addlen
            newmap[kp_x:kp_x+kernlen, kp_y:kp_y+kernlen] = kernmap
        orgmap = newmap[addlen:addlen+height, addlen:addlen+width]
        heatmaps[i] = orgmap
        # plt.imshow(heatmaps[i], interpolation='none')
        # plt.close('all')
    # timer.toc()
    # print ('getHeatmapBygkern took {:.3f}s').format(timer.total_time)
    return heatmaps



if __name__ == '__main__':
    # generate train test
    # trainPath = '/home/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/MOONdevkit/data/ImageSets/Main/train.txt'
    # testPath = '/home/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/data/MOONdevkit/data/ImageSets/Main/test.txt'
    # trains, tests = asign_train_test(324)
    # with open(trainPath, 'w') as f:
    #     for train in trains:
    #         f.write(train + '\n')
    # with open(testPath, 'w') as f:
    #     for test in tests:
    #         f.write(test + '\n')
    # print 'done'

    # rename image
    # rename_files('/home/cyl/py-faster-rcnn-test/data/MOONdevkit/data/JPEGImages')

    # gkern test
    # plt.figure()
    # plt.imshow(gkern(33), interpolation='none')
    # plt.close('all')

    # getHeatmapBygkern test
    # testmaps = np.zeros([68, 120, 120])
    # testmaps[0, 4, 4] = 1
    # testmaps[1, 115, 115] = 1
    # testmaps[2, 115, 4] = 1
    # testmaps[3, 4, 115] = 1
    # testmaps[4, 60, 60] = 1
    # getHeatmapBygkern(testmaps)

    # voc_ap test
    precision = [0.75, 0.8, 0.6, 0.7, 0.32]
    recall = [0.35, 0.1, 0.6, 0.6, 0.82]
    voc_ap(recall, precision)