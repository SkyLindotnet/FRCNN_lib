# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
from collections import OrderedDict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# config visual loss
__C.TRAIN.LOSSLIST = []

# transform image path
__C.TRANSFORM_IMPATH_roid = 0
__C.TRANSFORM_IMPATH = 0
__C.TRANSFORM_TARGET_DIR = '/data5/dataset/MulSrcData/300w-c1/trainval/images/'

# transform gt_kp into gt_box
__C.TRANSFORM_KP_TO_BOX = 0
__C.FILTER_INVALID_BOX = 1
__C.WIDER_FACE_STYLE = 1  # 1(for 68) 2(for 19)
__C.CLIP_BOXES = 1

# FITTING
__C.KP_ENABLE_FITTING = 0
__C.KP_FITTING_MODEL = 'AAM'
__C.KP_FITTING_DETDIR = ''

# RON
__C.ENABLE_RON = 0
if __C.ENABLE_RON:
    __C.TRAIN.NOBORDER = 0
    __C.TRAIN.PROB = 0.8  # 0.8 0.7
    __C.TRAIN.RON_MIN_SIZE = 10  # 10
    __C.TRAIN.ENABLE_NMS = 1  # 1
    __C.TRAIN.NMS = 0.7  # (0.7) 0.3
    __C.TRAIN.PRE_RON_NMS_TOP_N = 300  # 300
    __C.TRAIN.RON_NMS_TOP_N = 300  # if ENABLE_NMS
    __C.GENERATION_ANCHOR_RATIOS = [0.5, 1, 2]  # [1]
    # __C.MULTI_SCALE_RPN_NO = ['6']
    # __C.USED_RPN_NO = ['6']
    # __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[8, 16]]

    # __C.MULTI_SCALE_RPN_NO = ['6', '5']
    # __C.USED_RPN_NO = ['6', '5']
    # __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[8, 16], [4, 8]]

    __C.MULTI_SCALE_RPN_NO = ['6', '5', '4']
    __C.USED_RPN_NO = ['6', '5', '4']
    __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[8, 16], [4, 8], [2, 4]]

    # __C.MULTI_SCALE_RPN_NO = ['5', '4', '3']
    # __C.USED_RPN_NO = ['5', '4', '3']
    # __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[16, 32], [8, 16], [4, 8]]


# multi-scale RPN
__C.TRAIN.MULTI_SCALE_RPN = 0
if __C.TRAIN.MULTI_SCALE_RPN:
    __C.TRAIN.MULTI_SCALE_RPN_NUM = 2  # 2 3
    # __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[16, 32], [8, 16], [4, 8]]
    __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[8, 16, 32], [2, 4, 8]]
    # __C.TRAIN.MULTI_SCALE_RPN_STRIDE = [16, 8, 4]
    __C.TRAIN.MULTI_SCALE_RPN_STRIDE = [16, 8]
    __C.TRAIN.MULTI_SCALE_RPN_SELECT = 2  # 0:default 1:exclude un-matched gt 2: include un-matched gt by adapted anchor
    __C.TRAIN.MULTI_SCALE_RPN_POSITIVE_OVERLAP = 0.5
    __C.TRAIN.MULTI_SCALE_RPN_NEGATIVE_OVERLAP = 0.3
    # __C.TRAIN.MULTI_SCALE_RPN_PRE_NMS_TOP_Ns = [12000, 12000, 12000]
    # __C.TRAIN.MULTI_SCALE_RPN_POST_NMS_TOP_Ns = [2000, 2000, 2000]
    __C.TRAIN.MULTI_SCALE_RPN_PRE_NMS_TOP_Ns = [12000, 12000]
    __C.TRAIN.MULTI_SCALE_RPN_POST_NMS_TOP_Ns = [2000, 2000]

# shuffle heterology data
__C.TRAIN.shuffle_heterology = 0
# __C.TRAIN.shuffle_heterologyList = [['afw', 'helen', 'lfpw', 'ibug'], ['voc_wider'], ['morph']]
__C.TRAIN.shuffle_heterologyList = [['afw', 'helen', 'lfpw', 'ibug'], ['frgc']]
# [['afw', 'helen', 'lfpw', 'ibug'], ['voc_wider'], ['morph']]
# num of ethnicity
__C.TRAIN.ETHNICITY_NUM = 3

# Frozen Traing when training for proposal layer
__C.TRAIN.FrozenTraing = False

# cacade fc for proposal_target layer and train.py
__C.TRAIN.CacadeFC = True  # True False

# proposal Train like RPN TEST
__C.TRAIN.Frozen_NMS = False
__C.TRAIN.Frozen_NMS_THRESH = 0.7  # 0.7
__C.TRAIN.Frozen_PRE_NMS_TOP_N = 300  # 6000
__C.TRAIN.Frozen_POST_NMS_TOP_N = 300  # 300
__C.TRAIN.Frozen_MIN_SIZE = 0.1  # 0.1

# proposal target like FC TEST
__C.TRAIN.Frozen_BATCH_SIZE = 128  # 128
__C.TRAIN.Frozen_FG_THRESH = 0.7  # 0.5 300w(0.7(best) 0.8 0.6) fp3 hd1 hd1_2(0.5)
__C.TRAIN.Frozen_FG_FRACTION = 0.25  # 0.25 1

# load pre-trained model twice
__C.TRAIN.LOAD_MODEL_TWICE = False
__C.TRAIN.LOAD_MODEL_TWICE_PATH = ''

# multi-scale training
__C.TRAIN.MULTI_SCALE = False
__C.TRAIN.MULTI_SCALE_LIST = [0.5, 1.5]
__C.TRAIN.MULTI_SCALE_MIN = 600
__C.TRAIN.MULTI_SCALE_MAX = 1000

# use solveState during train
__C.TRAIN.WITH_SOLVERSTATE = False

# use during test
__C.TRAIN.TestDataSet = 'wider'  # threeHusFace wider facePlus
__C.TRAIN.ValImgList = ''
if __C.TRAIN.TestDataSet == 'wider':
    __C.TRAIN.TestMetrics = ['det_AP']
elif __C.TRAIN.TestDataSet == 'threeHusFace':
    __C.TRAIN.TestMetrics = ['all_cer_68']
elif __C.TRAIN.TestDataSet == 'Face_Plus':
    __C.TRAIN.TestMetrics = ['det_AP', 'det_AVE_KP_CER']
elif __C.TRAIN.TestDataSet == 'widerThusFace':
    __C.TRAIN.TestMetrics = ['det_AP', 'det_AVE_KP_CER']

# use SHARE LAYER COPY
__C.TRAIN.USE_SHARE_LAYER_COPY = 0
# __C.TRAIN.SOURCE_LAYERLIST = ['fc6', 'fc7', 'conv5_1', 'conv5_2', 'conv5_3']
__C.TRAIN.SOURCE_LAYERLIST = ['fc6', 'fc7']
# __C.TRAIN.SOURCE_LAYERLIST = ['dim_reduce']
# __C.TRAIN.SOURCE_LAYERLIST = ['fc7'] '_age', '_gender', '_ethnicity'
# __C.TRAIN.TARGET_LAYERL_SUFFIX = ['_keyPoint', '_attribute']  # TODO SUFFIX
__C.TRAIN.TARGET_LAYERL_SUFFIX = ['_kp']
# __C.TRAIN.TARGET_LAYERL_SUFFIX = ['_attribute']
# __C.TRAIN.TARGET_LAYERL_SUFFIX = ['_kp_attr']
# __C.TRAIN.TARGET_LAYERL_SUFFIX = ['_keyPoint']
# __C.TRAIN.TARGET_LAYERL_SUFFIX = ['_fc']

# use attributes from roid
__C.TRAIN.USE_ATTRIBUTE = 1  # 0
# __C.TRAIN.ATTRIBUTES = {'gt_age': 1, 'gt_gender': 1, 'gt_ethnicity': 1, 'gt_keyPoints': 83}
__C.TRAIN.KP = 5  # 1:all 2:inner 3:outer 4:eye 5:68 point
__C.TRAIN.VISUAL_ATTRIBUTES = 0
__C.TRAIN.VISUAL_SCALE = 1
__C.TRAIN.RPN_KP_REGRESSION = 0
# enable kp map
__C.TRAIN.PREDICE_KP_MAP = 1
__C.TRAIN.PREDICE_KP_REGRESSION = 0  # 1:reg(68) 0:cls(69)
__C.TRAIN.KP_REGRESSION_Gaussian = 1
__C.TRAIN.Gaussian_kernlen = 33  # 45 | 39 | 21 | 15 | 11 5 31 9 | 41 11 21 33
__C.TRAIN.Gaussian_sigma = 5  # 7 | 6 | 3 | 2 | 1 | 5

__C.TRAIN.KP_MAP_SAMPLE = 1  # 1:random sample 2: surround sample 0
__C.TRAIN.KP_MAP_FG_FRACTION = 0.5  # 1:3 0.25 0.5(best) 0.75 0.95
__C.TRAIN.MAP_offset = 1.3  # 1.5 1.3(o) 1.4 1.2(bit raise)
__C.TRAIN.MAP_ROI_offset = 1  # 0.9 1.3 1
__C.TRAIN.MAP_SIZE = 120  # 120 105 135(bit raise)
__C.TRAIN.MAP_ONLY_FG = 0
__C.TRAIN.WITH_BF_MAP = 0

# __C.TRAIN.KP_MAP_OHEM = 1

if __C.TRAIN.PREDICE_KP_MAP:
    __C.TRAIN.Frozen_PRE_NMS_TOP_N = 300  # 300 6000
    __C.TRAIN.Frozen_POST_NMS_TOP_N = 100  # 300
    __C.TRAIN.Frozen_BATCH_SIZE = 4  # 2 4 8 16(best)
    __C.TRAIN.Frozen_FG_FRACTION = 1

if __C.TRAIN.KP == 1:
    __C.TRAIN.ATTRIBUTES = {'gt_keyPoints': 166}  # 83*2
elif __C.TRAIN.KP == 2:
    __C.TRAIN.ATTRIBUTES = {'gt_keyPoints': 128}  # 64*2
elif __C.TRAIN.KP == 3:
    __C.TRAIN.ATTRIBUTES = [{'gt_keyPoints': 38}]  # 19*2
elif __C.TRAIN.KP == 4:
    __C.TRAIN.ATTRIBUTES = {'gt_keyPoints': 42}  # 21*2
elif __C.TRAIN.KP == 5:
    __C.TRAIN.ATTRIBUTES = [{'gt_keyPoints': 136}]  # 68*2
elif __C.TRAIN.KP == 6:
    if __C.TRAIN.ETHNICITY_NUM == 2:
        __C.TRAIN.ATTRIBUTES = [{'gt_keyPoints': 136}, {'gt_ages': 101},
                                {'gt_genders': 2}, {'gt_ethnicity': 2}]
    else:
        __C.TRAIN.ATTRIBUTES = [{'gt_keyPoints': 136}, {'gt_ages': 101},
                                {'gt_genders': 2}, {'gt_ethnicity': 3}]
elif __C.TRAIN.KP == 7:
    if __C.TRAIN.ETHNICITY_NUM == 2:
        __C.TRAIN.ATTRIBUTES = [{'gt_ages': 101}, {'gt_genders': 2},
                                {'gt_ethnicity': 2}]
    else:
        __C.TRAIN.ATTRIBUTES = [{'gt_ages': 101}, {'gt_genders': 2},
                                {'gt_ethnicity': 3}]
# elif __C.TRAIN.KP == 8:
#     __C.TRAIN.ATTRIBUTES = [{'gt_kp_map': 69}]

# run mixed pyramid
__C.MIXED_PYRAMID_MORE = False  # False
__C.MIXED_PYRAMID_NUM = 2

# run rpn pyramid
__C.RPN_PYRAMID_MORE = False  # False
__C.RPN_PYRAMID_NUM = 3

# run feature pyramid
__C.PYRAMID_MORE = False  # False
__C.PYRAMID_ONEFC = False  # False
# __C.PYRAMID_MORE_ANCHORS = [[4, 8], [16, 32]]  # v1
__C.PYRAMID_MORE_ANCHORS = [[1, 2, 4], [8, 16, 32]]  # v2
# __C.PYRAMID_MORE_ANCHORS = [[2, 4, 8], [8, 16, 32]]  # v2-1 v2-2
# __C.PYRAMID_MORE_ANCHORS = [[1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32]]  # v3
# __C.PYRAMID_MORE_ANCHORS = [[1, 2], [4, 8], [16, 32]]  # v4
# DO WORK DURING DEPLOYING
__C.PYRAMID_DETECT_DEBUG = 0
__C.PYRAMID_DETECT_NUM = 300
__C.PYRAMID_DETECT_INDEX = 1

# set allowed_border of anchors during training rpn
__C.TRAIN.RPN_ALLOWED_BORDER = 0  # 0 9999

# Parameters for "Online Hard-example Mining Algorithm" -add
__C.TRAIN.USE_OHEM = False
__C.TRAIN.USE_ROHEM = False
__C.TRAIN.USE_ROHEM_BATCHSIZE = 128  # batch size
################################
__C.TRAIN.OHEM_BY_LOSS = False
__C.TRAIN.OHEM_USE_BOX_LOSS = False
__C.TRAIN.OHEM_USE_NMS = False
__C.TRAIN.OHEM_NMS_THRESH = 0.7
# OHEM MINING BY SCORE
__C.TRAIN.FC_OHEM_MINING_FG = True
__C.TRAIN.FC_OHEM_MINING_BG = True
__C.TRAIN.FC_OHEM_MINING_BALANCE = 3
# when MINING_BALANCE != 1 or 2
__C.TRAIN.FC_OHEM_MINING_FG_threshold = 0.99
__C.TRAIN.FC_OHEM_MINING_BG_threshold = 0.1  # 0.3

# use extend anchors selecting -add
__C.TRAIN.EXTEND_ANCHORS_SELECTING = 0  # 1
__C.TRAIN.SELECTING_TOP_NUM = 3
__C.TRAIN.FUSE_ANCHORS_STRATEGY = False  # False
__C.TRAIN.FUSE_ANCHORS_THRESH = 0.5
__C.TRAIN.RPN_POSITION_ANCHORS_NUM_ADAPT = False  # True

# use fg/bg sample mining for rpn -add
__C.TRAIN.RPN_POSITION_ANCHORS_OHEM = False
__C.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_FG = False
__C.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_FG_threshold = 0.99
__C.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_BG = True
__C.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_BALANCE = 3
__C.TRAIN.RPN_POSITION_ANCHORS_OHEM_MINING_BG_threshold = 0.3

# use adapted num of position rois -add
__C.TRAIN.POSITION_ROIS_NUM_ADAPT = False  # True

# visual anchor during training -add
__C.TRAIN.VISUAL_ANCHORS = 0
__C.TRAIN.VISUAL_ANCHORS_IMG = ''
__C.TRAIN.VISUAL_ANCHORS_IMG_SCALE = 1
__C.TRAIN.VISUAL_ANCHORS_IMG_Flipped = False
__C.TRAIN.USING_GT = 0
__C.TRAIN.ANNOINFOS = None
__C.KP_UNDETECTED_FILL = 0
__C.KP_UNDETECTED_FILL_THRESH = 0.8
__C.KP_ADAPTED_MATCH = 0

# enable test with valid kp
__C.KP_VIS_KP_TEST = 0
__C.KP_VIS_KP_TEST_MASK = {}
__C.wo = 0

__C.KP_GT_MATCH_STRATEGY = 1  # 1:max kp num 2:max box overlap
__C.KP_GT_MATCH_NUM = 20
__C.KP_GT_MATCH_OVERLAP = 0.5
__C.KP_GT_MATCH_COMPLETION = 0
__C.KP_GT_MATCH_COMPLETION_PATH = ''

# config multiple anchors during training -add
__C.MORE_ANCHOR_v1 = 0
__C.MORE_ANCHOR_v2 = 0
__C.MORE_ANCHOR_v3 = 1
__C.MORE_ANCHOR_v4 = 0
__C.MORE_ANCHOR_v5 = 0

if __C.MORE_ANCHOR_v1:
    __C.DEFALU_ANCHOR_SCALES = (4, 8, 16, 32)
elif __C.MORE_ANCHOR_v2:
    __C.DEFALU_ANCHOR_SCALES = (2, 4, 8, 16, 32)
elif __C.MORE_ANCHOR_v3:
    __C.DEFALU_ANCHOR_SCALES = (1, 2, 4, 8, 16, 32)
elif __C.MORE_ANCHOR_v4:
    __C.DEFALU_ANCHOR_SCALES = (0.5, 1, 2, 4, 8, 16, 32)  # base_size: 4; min_rpn_size: 0.1
elif __C.MORE_ANCHOR_v5:
    __C.DEFALU_ANCHOR_SCALES = np.arange(1, 64, 2)  # base_size: 4; min_rpn_size: 0.1
else:
    __C.DEFALU_ANCHOR_SCALES = (8, 16, 32)

assert sum([__C.MORE_ANCHOR_v1, __C.MORE_ANCHOR_v2, __C.MORE_ANCHOR_v3, __C.MORE_ANCHOR_v4]) <= 1

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

# Iterations between metric -add
__C.TRAIN.METRIC_ITERS = 1000
__C.TRAIN.METRIC_RPN = False

# Iterations between metric -add
__C.TRAIN.METRIC_FILE = None

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

__C.TRAIN.RPN_NORMALIZE_TARGETS = False
__C.TRAIN.RPN_NORMALIZE_MEANS = None
__C.TRAIN.RPN_NORMALIZE_STDS = None

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'selective_search'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = False
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16

# config RPN_MIN_SIZE during training for adapting smaller anchors -add
if __C.MORE_ANCHOR_v1 or __C.MORE_ANCHOR_v2:
    __C.TRAIN.RPN_MIN_SIZE = 3  # 16
elif __C.MORE_ANCHOR_v3 or __C.MORE_ANCHOR_v4:
    __C.TRAIN.RPN_MIN_SIZE = 0.1  # 0.1
else:
    __C.TRAIN.RPN_MIN_SIZE = 16  # 16

# Proposal_RE TODO for train
__C.TRAIN.RE_PRE_NMS_TOP_N = 12000
__C.TRAIN.RE_POST_NMS_TOP_N = 2000
__C.TRAIN.RE_MIN_SIZE = 16
if __C.MORE_ANCHOR_v1 or __C.MORE_ANCHOR_v2:
    __C.TRAIN.RPN_MIN_SIZE = 3  # 16
elif __C.MORE_ANCHOR_v3 or __C.MORE_ANCHOR_v4:
    __C.TRAIN.RPN_MIN_SIZE = 0.1  # 0.1
else:
    __C.TRAIN.RPN_MIN_SIZE = 16  # 16

# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# whether use class aware box or not
__C.TRAIN.AGNOSTIC = False

#
# Testing options
#

__C.TEST = edict()

# debug
__C.TEST.DEBUG_IMPATH = ''
__C.TEST.USING_GT = 0

# RON Setting
if __C.ENABLE_RON:
    __C.TEST.BATCH_SIZE = 1
    __C.TEST.PROB = 0.8  # 0.03
    __C.TEST.RON_MIN_SIZE = 10  # 10
    __C.TEST.ENABLE_NMS = 1
    __C.TEST.NMS = 0.3  #
    __C.TEST.PRE_RON_NMS_TOP_N = 300  # 300
    __C.TEST.RON_NMS_TOP_N = 10  # if ENABLE_NMS
    __C.TEST.RON_SCALES = (320,)  # 320 480 540 640

# image save dir
__C.TEST.IMAGE_SAVE_DIR = ''

# image rotate test
__C.TEST.IMAGE_ROTATE = 0
__C.TEST.IMAGE_ROTATE_DEGREE = 90

# proposal Test
__C.TEST.Frozen_NMS = False
__C.TEST.Frozen_NMS_THRESH = 0.7  # 0.7
__C.TEST.Frozen_PRE_NMS_TOP_N = 300  # 6000
__C.TEST.Frozen_POST_NMS_TOP_N = 300  # 300
__C.TEST.Frozen_MIN_SIZE = 0.1  # 0.1
if __C.TRAIN.PREDICE_KP_MAP:
    __C.TEST.Frozen_PRE_NMS_TOP_N = 10  # 1 10 100 300 25
    __C.TEST.Frozen_POST_NMS_TOP_N = 100  # 300

# multi-scale testing
__C.TEST.MULTI_SCALE = True
__C.TEST.MULTI_SCALE_VALUE = 1  # 0.5, 2
__C.TEST.UNSCALE = False

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)  # 600

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000  # 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Visual Feature --add
__C.TEST.VISUAL_FEATURE = 0
__C.TEST.VISUAL_FEATURE_LAYER = 'conv1'
__C.TEST.PRINT_FUSE_WEIGHT = 0
__C.TEST.PRINT_FUSE_LAYER = 'dim_reduce'
__C.TEST.PRINT_FUSE_LAYER_CHANNELS = [256, 512, 512]

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'selective_search'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16
# enable proposal filter (keep max size)
__C.RPN_FILTER = 0
__C.RPN_FILTER_thresh = 0.9

# multi-scale RPN
# __C.TEST.MULTI_SCALE_RPN_PRE_NMS_TOP_Ns = [6000, 6000, 6000]
# __C.TEST.MULTI_SCALE_RPN_POST_NMS_TOP_Ns = [300, 300, 300]
__C.TEST.MULTI_SCALE_RPN_PRE_NMS_TOP_Ns = [6000, 6000]
__C.TEST.MULTI_SCALE_RPN_POST_NMS_TOP_Ns = [300, 300]

# config RPN_MIN_SIZE during testing for adapting smaller anchors -add
if __C.MORE_ANCHOR_v1 or __C.MORE_ANCHOR_v2:
    __C.TEST.RPN_MIN_SIZE = 3  # 16
    print "MORE"
elif __C.MORE_ANCHOR_v3 or __C.MORE_ANCHOR_v4:
    __C.TEST.RPN_MIN_SIZE = 0.1  # 0.1
    print "EXTREME"
else:
    __C.TEST.RPN_MIN_SIZE = 16  # 16
    print "DEFAULT"

## NMS threshold used on Proposal_RE layer
__C.TEST.RE_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RE_PRE_NMS_TOP_N = 300
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RE_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RE_MIN_SIZE = 16

# config RPN_MIN_SIZE during testing for adapting smaller anchors -add
if __C.MORE_ANCHOR_v1 or __C.MORE_ANCHOR_v2:
    __C.TEST.RE_MIN_SIZE = 3  # 16
    print "MORE"
elif __C.MORE_ANCHOR_v3 or __C.MORE_ANCHOR_v4:
    __C.TEST.RE_MIN_SIZE = 0.1  # 0.1
    print "EXTREME"
else:
    __C.TEST.RE_MIN_SIZE = 16  # 16
    print "DEFAULT"


# whether use class aware box or not
__C.TEST.AGNOSTIC = False

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'pascal_voc'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
