import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import math
import numpy as np
import caffe, shutil, os, cv2
import linecache

# release version

def visual_result(oriImgInfos, detImgInfos, detImgDir, imageSaveDir):
    mk_dir(imageSaveDir)
    for index, im_name in enumerate(oriImgInfos):
        # plot annotation
        annoImgInfo = oriImgInfos[im_name]
        im = cv2.imread(detImgDir + im_name+'.jpg')
        annoImgBoxs = np.array([annoImgBox.split(' ')[:5] for annoImgBox in annoImgInfo], dtype=np.float32)
        for i in range(len(annoImgBoxs)):
            major_axis_radius = annoImgBoxs[i][0]
            minor_axis_radius = annoImgBoxs[i][1]
            angle = (-1)*annoImgBoxs[i][2]*10
            center_x = annoImgBoxs[i][3]
            center_y = annoImgBoxs[i][4]
            cv2.ellipse(im, (center_x, center_y), (minor_axis_radius, major_axis_radius), angle, 0, 360, (0, 255, 0), 3)
        # plot detection
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
            cv2.putText(im, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
            cv2.putText(im, str(score), (x1, int(y2+15)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
            imfile = imageSaveDir + '/re_' + '_'.join(im_name.split('/')) + '.jpg'
            cv2.imwrite(imfile, im)
    pass

def loadResultFile(filePath, im_names):
    # load annotated image
    with open(filePath, 'r') as f:
        lines = f.readlines()
    index = 0
    imgsInfo = {}
    # im_names = linecache.getlines(testImgList)
    for i in range(len(im_names)):
        im = im_names[i][:-1]
        aim = lines[index][:-1]
        assert im == aim, "image is not matched"
        index = index + 1
        roiNum = int(lines[index][:-1])
        imgsInfo[aim] = [lines[j][:-1] for j in range(index+1, index+1 + roiNum)]
        index = index + roiNum + 1
    return imgsInfo

def regress_match(oriImgInfos, detImgInfos, im_names, matchSavePath, detImgDir, overthresh = 0.5):
    #prepare to make annotated image
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
            angle = (-1)*float(oriImgBoxs[ori_i][2])*10
            center_x = int(float(oriImgBoxs[ori_i][3]))
            center_y = int(float(oriImgBoxs[ori_i][4]))
            oriArea = math.pi * float(oriImgBoxs[ori_i][0]) * float(oriImgBoxs[ori_i][1])
            oriInfos[im_name].append([major_axis_radius, minor_axis_radius, angle, center_x, center_y, oriArea])
        oriDet[im_name] = [False] * len(oriImgBoxs)
        oriRoiSum += len(oriImgBoxs)

    #sorted detImgInfos
    detScores = []
    detBoxes = []
    detImNames = []
    for i in range(len(im_names)):
        im_name = im_names[i][:-1]
        detScores.extend([float(detImgBox.split(' ')[-1]) for detImgBox in detImgInfos[im_name]])
        detBoxe = [np.array(detImgBox.split(' ')[:4]).astype(float) for detImgBox in detImgInfos[im_name]]
        detBoxes.extend(detBoxe)
        detNum = len(detImgInfos[im_name])
        detImNames.extend([im_name]*detNum)
    sorted_ind = np.argsort(-np.array(detScores))
    sorted_scores = (-1) * np.sort(-np.array(detScores))
    sorted_detImNames = np.array([detImNames[i] for i in sorted_ind])
    sorted_detBoxes = [detBoxes[i] for i in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(sorted_detImNames)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # record match results
    matchSaveFile = open(matchSavePath, 'w')
    for d in range(nd):
        sorted_detBox = sorted_detBoxes[d]
        sorted_detBox_str = ' '.join(sorted_detBox.astype(np.str))
        det_imgName = sorted_detImNames[d]
        det_scores = sorted_scores[d]
        det_sx = int(sorted_detBox[0])
        det_sy = int(sorted_detBox[1])
        det_w = sorted_detBox[2]
        det_h = sorted_detBox[3]
        det_ex = int(det_sx + det_w)
        det_ey = int(det_sy + det_h)
        det_area = (det_w + 1.) * (det_h + 1.)
        annoIm = cv2.imread(detImgDir + det_imgName+'.jpg')
        oriInfo = oriInfos[det_imgName]
        det_overlaps = []

        for ori_i in range(len(oriInfo)):
            [major_axis_radius, minor_axis_radius, angle, center_x, center_y, oriArea] = oriInfo[ori_i]
            cv2.ellipse(annoIm, (center_x, center_y), (minor_axis_radius, major_axis_radius), angle, 0, 360, (0, 255, ori_i), -3)
            det_reg = annoIm[det_sy:det_ey+1, det_sx:det_ex+1, :].reshape((det_ey-det_sy+1)*(det_ex-det_sx+1), 3)
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
                oriImgBox_str = oriImgInfos[det_imgName][overmax_i][:-2]
                matchSaveFile.write('%s %f %s %s\n' % (det_imgName, det_scores, sorted_detBox_str, oriImgBox_str))
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    matchSaveFile.close()

def regress_det_model(net, im_names, resultFilePath, detImgDir, recompute=0,
                      CONF_THRESH=0.8, NMS_THRESH=0.3):
    if not os.path.exists(resultFilePath) or recompute:
        resultFile = open(resultFilePath, 'w')
        for im_name in im_names:
            image_name = im_name[:-1]
            im_path = os.path.join(detImgDir, image_name + '.jpg')
            im = cv2.imread(im_path)

            # Detect all object classes and regress object bounds
            scores, boxes = im_detect(net, im)
            cls_ind = 1  # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            # record result
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            resultFile.write(image_name+'\n')
            resultFile.write(str(len(inds))+'\n')
            if len(inds) == 0:
                 continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                resultFile.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2]-bbox[0])+' '+
                                    str(bbox[3]-bbox[1])+' '+str(score)+'\n')
        resultFile.close()

def regress_sampler(methodName, prototxt, modelPath, detFilePath, annoFilePath, saveDirPath, detImgDir, visual=1):
    # initialize parameters of net and load net
    gpu_id = 0
    cfg.TEST.HAS_RPN = True
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)

    # detect images and record results
    im_names = linecache.getlines(detFilePath)
    mk_dir(saveDirPath)
    resultSavePath = os.path.join(saveDirPath, '%s_det_fddb.txt' % methodName)
    regress_det_model(net, im_names, resultSavePath, detImgDir)

    # match gt and record results
    detInfos = loadResultFile(resultSavePath, im_names)
    annoInfos = loadResultFile(annoFilePath, im_names)

    if visual:
        imageSaveDir = os.path.join(saveDirPath, '%s_det_img' % methodName)
        visual_result(annoInfos, detInfos, detImgDir, imageSaveDir)

    matchSavePath = os.path.join(saveDirPath, '%s_match_fddb.txt' % methodName)
    regress_match(annoInfos, detInfos, im_names, matchSavePath, detImgDir)

    print  'done'

def mk_dir(saveImgDir, recompute=0):
    if not os.path.exists(saveImgDir):
        os.makedirs(saveImgDir)
    elif recompute:
        shutil.rmtree(saveImgDir)
        os.makedirs(saveImgDir)

if __name__ == '__main__':
    # test example
    prototxt = cfg.ROOT_DIR + '/models/wider_face/VGG16/faster_rcnn_end2end/test_fuse_multianchor_v2.prototxt'
    modelPath = cfg.ROOT_DIR + '/output/wider_face/model/VGG16_faster_rcnn_end2end_with_fuse_multianchor_iter_100000_v2.caffemodel'
    detImgDir = cfg.ROOT_DIR + '/data/DB/face/FDDB/'
    detFilePath = cfg.ROOT_DIR + '/data/DB/face/FDDB/Annotations_new_v2/FDDB-fold-all.txt'
    annoFilePath = cfg.ROOT_DIR + '/data/DB/face/FDDB/Annotations_new_v2/FDDB-ellipseList-all.txt'
    saveDirPath = cfg.ROOT_DIR + '/data/DB/face/FDDB/match'

    methodName = 'FMA_v2'
    regress_sampler(methodName, prototxt, modelPath, detFilePath, annoFilePath, saveDirPath, detImgDir)