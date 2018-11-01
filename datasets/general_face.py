# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
# from utils.imdb_explore.FaceImage import FaceImage
from utils.MultiAttributeDB.FaceImage import FaceImage
# from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle

class general_face(imdb):
    def __init__(self, name, image_set, data_path):
        imdb.__init__(self, name)
        self._image_set = image_set
        self._data_path = data_path
        self._classes = ('__background__',  # always index 0
                         'face')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        # assert os.path.exists(self._devkit_path), \
        #         'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def visual_attribute(self, im_path, scale, gt_boxes, gt_keyPoints, Flipped=True,
                         displayRegion=None):
        plt.figure()
        ax = plt.subplot(1, 2, 1)
        im = cv2.imread(im_path)
        if Flipped:
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
        if displayRegion is not None:
            display_keyPoints = gt_keyPoints[:, displayRegion, :]
            display_keyPoints = display_keyPoints.reshape((-1, 2))
            plt.plot(display_keyPoints[:, 0], display_keyPoints[:, 1], 'go', ms=1.5, alpha=1)
        else:
            gt_keyPoints = gt_keyPoints.reshape((-1, 2))
            plt.plot(gt_keyPoints[:, 0], gt_keyPoints[:, 1], 'go', ms=1.5, alpha=1)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        fi = FaceImage(index)
        image_path = fi.image_path
        # image_path = os.path.join(self._data_path, 'JPEGImages',
        #                           index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            if cfg.TRANSFORM_IMPATH:
                image_index = []
                for c in f.readlines():
                    type_str = c.split('/')[4]
                    suffix = c.split('/')[-1].split('.')[-1]
                    imname = c.split('/')[-1].split('.')[0]
                    if type_str == 'afw':
                        imname = imname.split('_')[0]
                        impath = os.path.join(cfg.TRANSFORM_TARGET_DIR,
                                              'afw_%s_s1_r0_x0_y0.%s' % (imname, suffix))
                    elif type_str == '300w-c1':
                        im_type = imname.split('_')[0]
                        imdirlist = c.split('/')
                        imdirlist[4] = im_type
                        imdir = '/'.join(imdirlist[:7])
                        if im_type == 'afw':
                            imname = imname.split('_')[1]
                            i = 1
                            impath = os.path.join(imdir,
                                                  '%s_%d.%s' % (imname, i, suffix))
                            while not os.path.exists(impath.strip()):
                                i = i + 1
                                impath = os.path.join(imdir,
                                                      '%s_%d.%s' % (imname, i, suffix))
                        else:
                            imdirlist = imdir.split('/')
                            imdirlist[5] = 'trainset'
                            imdir = '/'.join(imdirlist)
                            imname = imname.split('_')[1] + '_' + imname.split('_')[2]
                            impath = os.path.join(imdir,
                                                  '%s.%s' % (imname, suffix))
                    else:
                        impath = os.path.join(cfg.TRANSFORM_TARGET_DIR,
                                              '%s_%s_s1_r0_x0_y0.%s' % (type_str, imname, suffix))
                    if not os.path.exists(impath.strip()):
                        break
                    image_index.append(impath.strip())
            else:
                rootdir = cfg.ROOT_DIR
                image_index = [os.path.join(rootdir, x.strip()) for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            gt_keyPoints = self.roidb[i]['gt_keyPoints'].copy()
            # if 1:
            #     im_path = self.image_path_at(i)
            #     self.visual_attribute(im_path, 1, boxes, gt_keyPoints, Flipped=False)

            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1

            for k in range(0, gt_keyPoints.shape[1], 2):
                if -1 in gt_keyPoints[:, k]:
                    print 's'
                oldKpX = gt_keyPoints[:, k].copy()
                gt_keyPoints[:, k] = widths[i] - oldKpX - 1

            # adjust label of keyPoint
            gt_keyPoints = gt_keyPoints.reshape(-1, gt_keyPoints.shape[1]/2, 2)
            LcheepMap = range(0, 8)
            RcheepMap = range(16, 8, -1)
            LbrowMap = range(17, 22)
            RbrowMap = range(26, 21, -1)
            LeyeMap = range(36, 42)
            ReyeMap = [45, 44, 43, 42, 47, 46]
            LnoseMap = range(31, 33)
            RnoseMap= range(35, 33, -1)
            LmouthMap = [48, 49, 50, 58, 59, 60, 61, 67]
            RmouthMap = [54, 53, 52, 56, 55, 64, 63, 65]
            LallMap = LcheepMap + LbrowMap + LeyeMap + LnoseMap + LmouthMap
            RallMap = RcheepMap + RbrowMap + ReyeMap + RnoseMap + RmouthMap

            Lall_keypoints = gt_keyPoints[:, LallMap, :].copy()
            gt_keyPoints[:, LallMap, :] = gt_keyPoints[:, RallMap, :]
            gt_keyPoints[:, RallMap, :] = Lall_keypoints
            gt_keyPoints = gt_keyPoints.reshape(-1, 136)
            # if 1:
            #     im_path = self.image_path_at(i)
            #     self.visual_attribute(im_path, 1, boxes, gt_keyPoints, Flipped=True)
                # for disRegion in [LcheepMap, LbrowMap, LeyeMap, LnoseMap, LmouthMap,
                #                   RcheepMap, RbrowMap, ReyeMap, RnoseMap, RmouthMap]:
                #     self.visual_attribute(im_path, 1, boxes, gt_keyPoints, Flipped=True,
                #                           displayRegion=disRegion)

            if not (boxes[:, 2] >= boxes[:, 0]).all():
                print 'done'
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'flipped': True,
                     'seg_areas': self.roidb[i]['seg_areas'],
                     'gt_ages': self.roidb[i]['gt_ages'],
                     'gt_genders': self.roidb[i]['gt_genders'],
                     'gt_ethnicity': self.roidb[i]['gt_ethnicity'],
                     'gt_keyPoints': gt_keyPoints,
                     'scale': 1}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def append_multiScaled_images(self):
        num_images = self.num_images
        for scale in cfg.TRAIN.MULTI_SCALE_LIST:
            for i in xrange(num_images):
                entry = {'boxes' : self.roidb[i]['boxes'],
                         'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                         'gt_classes' : self.roidb[i]['gt_classes'],
                         'flipped' : self.roidb[i]['flipped'],
                         'seg_areas': self.roidb[i]['seg_areas'],
                         'gt_ages': self.roidb[i]['gt_ages'],
                         'gt_genders': self.roidb[i]['gt_genders'],
                         'gt_ethnicity': self.roidb[i]['gt_ethnicity'],
                         'gt_keyPoints': self.roidb[i]['gt_keyPoints'],
                         'scale': scale}
                self.roidb.append(entry)
        self._image_index = self._image_index * (len(cfg.TRAIN.MULTI_SCALE_LIST) + 1)

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        fi = FaceImage(index)
        num_objs = len(fi.faces)
        objs = fi.faces

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ages = np.zeros((num_objs), dtype=np.float32)
        genders = np.zeros((num_objs), dtype=np.int32)
        ethnicity = np.zeros((num_objs), dtype=np.int32)
        kp_num = cfg.TRAIN.ATTRIBUTES[0]['gt_keyPoints']
        keyPoints = np.zeros((num_objs, kp_num), dtype=np.float32)  # sum of keyPoints is 83

        # Load object bounding boxes into a data frame.
        img = fi.get_opened_image()
        img_width = img.shape[1]
        img_height = img.shape[0]
        for ix, obj in enumerate(objs):
            x1, y1, x2, y2 = obj['face_rectangle_xyxy']
            # revise outlier
            x1 = min(max(0, x1), img_width-1)
            y1 = min(max(0, y1), img_height-1)
            x2 = min(max(0, x2), img_width-1)
            y2 = min(max(0, y2), img_height-1)

            if x1 > x2 or y1 > y2:
                print x1, y1, x2, y2
            if x1 == x2 or y1 == y2:
                print x1, y1, x2, y2
            cls = 1
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            # additional face attribute
            if obj['age'] is None:
                ages[ix] = -1
            else:
                ages[ix] = obj['age']

            if obj['gender'] is None:
                genders[ix] = -1
            elif obj['gender'] == 'Male':
                genders[ix] = 0
            elif obj['gender'] == 'Female':
                genders[ix] = 1
            else:
                print 'format of gender is error'
                exit(0)
            if cfg.TRAIN.ETHNICITY_NUM == 2:
                if obj['ethnicity'] is None:
                    ethnicity[ix] = -1
                elif obj['ethnicity'] == 'White':
                    ethnicity[ix] = 0
                elif obj['ethnicity'] == 'Asian':
                    ethnicity[ix] = -1
                elif obj['ethnicity'] == 'Black':
                    ethnicity[ix] = 1
                else:
                    print 'format of ethnicity is error'
                    exit(0)
            else:
                if obj['ethnicity'] is None:
                    ethnicity[ix] = -1
                elif obj['ethnicity'] == 'White':
                    ethnicity[ix] = 0
                elif obj['ethnicity'] == 'Asian':
                    ethnicity[ix] = 2
                elif obj['ethnicity'] == 'Black':
                    ethnicity[ix] = 1
                else:
                    print 'format of ethnicity is error'
                    exit(0)

            # t2 'None':-1 'White':0 'Asian': 1 'Black': 2


            if obj['keypoints'] is None:
                keyPoints[ix, :] = 0
            else:
                keyPoints[ix, :] = obj['keypoints'].ravel()

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'gt_ages': ages,
                'gt_genders': genders,
                'gt_ethnicity': ethnicity,
                'gt_keyPoints': keyPoints,
                'scale': 1}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
