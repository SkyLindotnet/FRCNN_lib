import numpy as np
import os, sys
from collections import OrderedDict
from file_tool import *
from LandmarkTools import landmark_reduce
from pylab import *
from mylab.draw import *


def parse_det_str(det_str):
    # x y w h
    dets = np.array([float(i) for i in det_str.split(' ')]).reshape(-1, 5)
    scores = dets[:, -1]
    dets = dets[:, 0:4]
    dets[:, 2] += dets[:, 0]
    dets[:, 3] += dets[:, 1]
    return dets, scores


'''
According to the argument "dbtype"
return a corresponding converter
'''


def lbl_img_path_converter_factory(path, arg4version):
    responsibility_chain = [imdb_lbl_img_path_converter,
                            mulsrc_lbl_img_path_converter,
                            wider_lbl_img_path_converter,
                            facepp_lbl_img_path_converter]
    for rc in responsibility_chain:
        img_path, lbl_path = rc(path, arg4version)
        if img_path is not None: return img_path, lbl_path

    raise Exception("No converter accepts the path: '{}'.".format(path))


'''
imdb_lbl_img_path_converter
respondsible for:
    @imdb
    @imdb2

'''


def imdb_lbl_img_path_converter(path, version):
    if 'facepp' in path.lower(): return None, None
    if 'imdb' not in path.lower(): return None, None
    avaiable_versions = ['_label_agr', '_label_nose', '_label_washed', '_attributes']
    if version not in avaiable_versions: version = avaiable_versions[0]  # default use latest version

    if path.endswith('.jpg'):

        parts = path.split('/')
        if parts[-2] == 'ref':
            parts[-5] += version
        else:
            parts[-4] += version

        parts[-1] = parts[-1].replace('.jpg', '.txt')
        return path, '/'.join(parts)

    if path.endswith('.txt'):
        parts = path.split('/')
        for av in avaiable_versions:
            parts[-4] = parts[-4].replace(av, '')
        parts[-1] = parts[-1].replace('.txt', '.jpg')
        return '/'.join(parts), path

    raise Exception("path: '{}' ends with unexpected '{}'".format(path, path[-4:]))


'''
mulsrc_lbl_img_path_converter
respondsible for:
    @afw
    @helen
    @ibug
    @lfpw
    @morph
    @frgc
'''


def mulsrc_lbl_img_path_converter(path, version):
    mulsrc_setnames = ['afw', 'helen', 'ibug', 'lfpw', 'morph', 'frgc', '300w', 'aflw', 'cofw']
    check = False
    for n in mulsrc_setnames:
        if n.lower() in path.lower():
            check = True
            break
    if not check: return None, None

    avaiable_versions = ['labels']
    if version not in avaiable_versions: version = avaiable_versions[0]  # default use latest version

    suffix = '.' + path.split('.')[-1]

    if suffix == '.png' or suffix == '.jpg':  # image path
        sp = path.split('/')
        sp[-2] = version
        sp = '/'.join(sp)
        sp = sp.replace(suffix, '.txt')

        # return {img,lbl} path
        return path, sp

    elif path.endswith('.txt'):
        sp = path.split('/')
        sp[-2] = 'images'
        sp = '/'.join(sp)

        sp1 = sp.replace('.txt', '.jpg')
        sp2 = sp.replace('.txt', '.png')

        if os.path.exists(sp1):
            sp = sp1
        elif os.path.exists(sp2):
            sp = sp2
        else:
            raise Exception("'{}' and '{}' both not found.".format(sp1, sp2))

        # return {img,lbl} path
        return sp, path
    else:
        raise Exception("Failed to handle: '{}'".format(path))

'''
facepp_lbl_img_path_converter
respondsible for:
    @facepp
'''


def facepp_lbl_img_path_converter(path, version):
    mulsrc_setnames = ['facepp']
    check = False
    for n in mulsrc_setnames:
        if n.lower() in path.lower():
            check = True
            break
    if not check: return None, None

    avaiable_versions = ['labels']
    if version not in avaiable_versions: version = avaiable_versions[0]  # default use latest version

    suffix = '.' + path.split('.')[-1]

    if suffix == '.png' or suffix == '.jpg':  # image path
        sp = path.split('/')
        sp[-4] = version
        sp = '/'.join(sp)
        sp = sp.replace(suffix, '.txt')

        # return {img,lbl} path
        return path, sp

    elif path.endswith('.txt'):
        sp = path.split('/')
        sp[-4] = 'images'
        sp = '/'.join(sp)

        sp = sp.replace('.txt', '.jpg')

        return sp, path
    else:
        raise Exception("Failed to handle: '{}'".format(path))

def wider_lbl_img_path_converter(path, version):
    if 'voc_wider' not in path: return None, None
    avaiable_versions = ['Labels']
    if version not in avaiable_versions: version = avaiable_versions[0]  # default use latest version

    if path.endswith('.jpg'):
        lbl_path = path.replace('Images', version).replace('.jpg', '.txt')
        return path, lbl_path
    elif path.endswith('.txt'):
        img_path = path.replace(version, 'Images').replace('.txt', '.jpg')
        return img_path, path

    else:
        raise Exception("Failed to handle: '{}'".format(path))


class FaceImage():
    def vis(self):
        def __get_attr_str():
            ret_list = []
            ages = self.get_ages()
            genders = self.get_genders()
            races = self.get_ethnicity()
            for a, g, r in zip(ages, genders, races):
                if a is not None:
                    s = '{}\n{}\n{}'.format(a, g, r)
                    ret_list.append(s)
                else:
                    ret_list.append(None)
            return ret_list

        fig = figure(figsize=(14, 7))
        ax = fig.add_subplot(111)

        imshow(self.get_opened_image())
        boxes = self.get_bboxes('xyxy')
        kps = self.get_keypoints()
        draw_bbox(ax, boxes, color='cyan', extra_texts=__get_attr_str())
        for kp in kps:
            if kp is None: continue
            plot(kp[:, 0], kp[:, 1], 'g.', ms=5, alpha=0.8)
            for i, v in enumerate(kp):
                text(v[0], v[1], str(i+1), fontsize=8, color='cyan')
            # text(kp[:, 0], kp[:, 1], '1', fontsize=12)

        axis('off')
        print self.image_path
        show()

    def vis_test(self, vis_text=0, vis_rebox=1, vis_kp_style=1, vis_oribox=1):
        def __get_attr_str():
            ret_list = []
            ages = self.get_ages()
            genders = self.get_genders()
            races = self.get_ethnicity()
            for a, g, r in zip(ages, genders, races):
                if a is not None:
                    s = '{}\n{}\n{}'.format(a, g, r)
                    ret_list.append(s)
                else:
                    ret_list.append(None)
            return ret_list

        fig = figure(figsize=(14, 7))
        ax = fig.add_subplot(111)

        imshow(self.get_opened_image())
        boxes = self.get_bboxes('xyxy')
        kps = self.get_keypoints()
        if vis_oribox:
            draw_bbox(ax, boxes, color='blue', extra_texts=__get_attr_str())
        if vis_rebox:
            boxes = self.transform_kp_to_box(WIDER_FACE_STYLE=vis_kp_style)
            draw_bbox(ax, boxes, color='r', extra_texts=__get_attr_str(), ls='--')
        for kp in kps:
            if kp is None: continue
            plot(kp[:, 0], kp[:, 1], 'g.', ms=5, alpha=0.8)
            if vis_text:
                for i, v in enumerate(kp):
                    text(v[0], v[1], str(i+1), fontsize=8, color='cyan')

        axis('off')
        print self.image_path
        show()

    def transform_kp_to_box(self, fh=3, WIDER_FACE_STYLE=1):
        kp_num = self.get_keypoints()[0].shape[0]
        boxes = self.get_bboxes('xyxy')
        gt_keyPoints = self.get_keypoints()
        for box, gt_keyPoint in zip(boxes, gt_keyPoints):
            gt_keyPoint = gt_keyPoint.reshape([kp_num, 2])
            x1 = np.min(gt_keyPoint[:, 0])
            y1 = np.min(gt_keyPoint[:, 1])
            x2 = np.max(gt_keyPoint[:, 0])
            y2 = np.max(gt_keyPoint[:, 1])
            if WIDER_FACE_STYLE == 1:
                offset = (y2 - y1) / fh
                y1 = y1 - offset
            elif WIDER_FACE_STYLE == 2:
                if kp_num == 19:  # aflw-full
                    y_offset = (y2 - y1) / fh
                    y1 = y1 - y_offset
                    x_offset = (gt_keyPoint[1, 0] - gt_keyPoint[0, 0])  # 2, 1
                    x1 = x1 - x_offset
                    x_offset = (gt_keyPoint[5, 0] - gt_keyPoint[4, 0])  # 5, 4
                    x2 = x2 + x_offset
                elif kp_num == 29:  # cofw
                    y_offset = (y2 - y1) / fh
                    y1 = y1 - y_offset
                    x_offset = (gt_keyPoint[4, 0] - gt_keyPoint[0, 0])
                    x1 = x1 - x_offset
                    x_offset = (gt_keyPoint[1, 0] - gt_keyPoint[6, 0])
                    x2 = x2 + x_offset
            box[:] = np.hstack([x1, y1, x2, y2])
        # if cfg.CLIP_BOXES:
        #     im_shape = cv2.imread(im).shape[0:2]
        #     boxes = clip_boxes(boxes, im_shape)
        return boxes

    def revise_content(self, idx, D):
        '''use this function to revise old contents by those labeled by human'''

        contain_face = D['Contains_Face']
        box_precise = D['Box']
        kpt_precise = D['Keypoints']

        age = D['Age']
        race = D['Race']
        gender = D['Gender']

        if contain_face != 'True' or box_precise != 'Precise':
            self.faces[idx]['face_token'] = 'BAD'
            return
        if kpt_precise != 'Precise':
            self.faces[idx]['keypoints'] = None
            self.faces[idx]['keypoints83'] = None

        self.faces[idx]['age'] = age
        self.faces[idx]['ethnicity'] = race
        self.faces[idx]['gender'] = gender

    def write_revised_content(self, img_dir, lbl_dir):
        new_faces = []

        for f in self.faces:
            if f['face_token'] is not 'BAD': new_faces.append(f)
        self.faces = new_faces
        if 'imdb2/' in self.image_path:
            img_replace = '/home/ylxie/facepp/imdb2'
        else:
            img_replace = '/home/ylxie/facepp/imdb'
        imgpath = self.image_path.replace(img_replace, img_dir)
        lblpath = self.image_path.replace(img_replace, lbl_dir).replace('.jpg','.txt')
        assert 'NEW' in imgpath
        if not os.path.exists(os.path.dirname(imgpath)):
            os.makedirs(os.path.dirname(imgpath))



        os.system('cp {} {}'.format(self.image_path, imgpath))
        self.save_txt(lblpath)

    def replace_agr(self, face_idx, agr):
        assert face_idx < len(self.faces), 'index {} is OOB, face-ind-max is {}'.format(face_idx, len(self.faces))

        if None not in agr:  # only insert if none of these are None
            self.faces[face_idx]['age'] = agr[0]
            self.faces[face_idx]['gender'] = agr[1]
            self.faces[face_idx]['ethnicity'] = agr[2]

    def replace_nose(self, face_idx, nose_pt):

        assert face_idx < len(self.faces), 'index {} is OOB, face-ind-max is {}'.format(face_idx, len(self.faces))
        kpt68 = self.faces[face_idx]['keypoints']

        kpt68[27:31, :] = nose_pt
        self.faces[face_idx]['keypoints'] = kpt68

    def insert_newbox(self, det_str):
        from utils.cython_bbox import bbox_overlaps

        # inserted new det-box in form: x y w h s
        # compute overlap with previous box,
        # ____addnew____|__ignore__|_____replace______
        #               t1         t2
        #              0.05       0.5
        new_boxs, scores = parse_det_str(det_str)

        if len(self.faces) != 0:
            fpp_boxs = self.get_bboxes('xyxy')

            overlaps = bbox_overlaps(
                np.ascontiguousarray(new_boxs, dtype=np.float),
                np.ascontiguousarray(fpp_boxs, dtype=np.float))
            gt_assignment = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)

            replace_thres = 0.5  # >= 0.5
            replace_indces = np.where(max_overlaps >= replace_thres)[0]
            be_replaced_fpp_indces = gt_assignment[replace_indces]

            for be_ind, ind in zip(be_replaced_fpp_indces, replace_indces):
                b = new_boxs[ind, :]  # xyxy
                dic = self.faces[be_ind]
                dic['face_rectangle'] = {'height': b[3] - b[1], 'left': b[0], 'top': b[1], 'width': b[2] - b[0]}
                dic['face_rectangle_xyxy'] = b
                self.faces[be_ind] = dic

        new_added_thres = 0.05  # <=0.05
        new_added_indces = np.where(max_overlaps <= new_added_thres)[0]

        for i in new_added_indces:
            attr_dict = OrderedDict()

            attr_dict['age'] = None
            attr_dict['blurness'] = [None, None]
            attr_dict['ethnicity'] = None
            attr_dict['left_eye_status'] = [None, None, None, None, None, None]
            attr_dict['right_eye_status'] = [None, None, None, None, None, None]
            attr_dict['facequality'] = [None, None]
            attr_dict['gender'] = None
            attr_dict['glass'] = None
            attr_dict['headpose'] = [None, None, None]
            attr_dict['smile'] = [None, None]
            b = new_boxs[i, :]

            attr_dict['face_rectangle'] = {'height': b[3] - b[1], 'left': b[0], 'top': b[1], 'width': b[2] - b[0]}

            attr_dict['face_rectangle_xyxy'] = b
            attr_dict['face_token'] = None
            attr_dict['keypoints'] = None
            attr_dict['keypoints83'] = None

            self.faces.append(attr_dict)

            # ignored_indces = np.where((max_overlaps < replace_thres) & (max_overlaps > new_added_thres))[0]
            # ignored_fpp_indces = gt_assignment[ignored_indces]

    def __set_lbl_img_path(self, file_path, arg4version):
        img_path, lbl_path = lbl_img_path_converter_factory(file_path, arg4version)
        self.image_path = img_path
        self.label_path = lbl_path

    def get_opened_image(self):
        if self.opened_image is None:
            self.opened_image = imread(self.image_path)
            if self.opened_image.ndim == 2:
                [x, y] = self.opened_image.shape
                self.opened_image = self.opened_image.repeat(3, 1).reshape(x, y, 3)
        return self.opened_image

    def get_bboxes(self, order='xyxy'):
        ava_orders = ['whxy', 'xywh', 'xyxy']
        if order not in ava_orders:
            raise Exception('Bad order: {}, available orders are [{}]'.format(order, ', '.join(ava_orders)))

        ret = np.zeros(shape=(len(self.faces), 4))
        for i, f in enumerate(self.faces):
            xyxy = f['face_rectangle_xyxy']
            if order == 'xyxy':
                ret[i, :] = xyxy
            else:
                x1 = xyxy[0]
                y1 = xyxy[1]
                x2 = xyxy[2]
                y2 = xyxy[3]
                w = x2 - x1
                h = y2 - y1
            # covert from xyxy
            if order == 'whxy':

                ret[i, 0] = w
                ret[i, 1] = h
                ret[i, 2] = x1
                ret[i, 3] = y1
            elif order == 'xywh':
                ret[i, 0] = x1
                ret[i, 1] = y1
                ret[i, 2] = w
                ret[i, 3] = h

        return ret

    def get_genders(self):
        ret = []
        for f in self.faces:
            ret.append(f['gender'])
        return ret

    def get_ethnicity(self):
        ret = []
        for f in self.faces:
            ret.append(f['ethnicity'])
        return ret

    def get_ages(self):
        ret = []
        for f in self.faces:
            ret.append(f['age'])
        return ret

    def get_keypoints(self, with_83=False):
        ret = []
        for i, f in enumerate(self.faces):
            if with_83:
                ret.append(f['keypoints83'])
            else:
                ret.append(f['keypoints'])
        return ret

    def format_content(self):
        faces_str = []
        for i in range(self.get_bboxes().shape[0]):
            face = self.faces[i]
            face_str = []
            for k in face.keys():

                if k in ['keypoints83', 'face_rectangle_xyxy']: continue
                if k == 'keypoints':
                    keypoints68 = face['keypoints']
                    if keypoints68 is None:
                        face_str.append(None)
                    else:
                        for j in range(keypoints68.shape[0]):
                            face_str.append(keypoints68[j, 0])
                            face_str.append(keypoints68[j, 1])
                    continue
                if k == 'face_rectangle':
                    v = face[k]
                    face_str.append(v['height'])
                    face_str.append(v['left'])
                    face_str.append(v['top'])
                    face_str.append(v['width'])

                    continue
                v = face[k]
                if type(v) == OrderedDict:
                    for kk in v.keys():
                        vv = v[kk]
                        face_str.append(vv)
                elif type(v) == list:
                    for vv in v:
                        face_str.append(vv)
                else:
                    face_str.append(v)
            faces_str.append(face_str)

        for i in range(len(faces_str)):
            faces_str[i] = [str(x) for x in faces_str[i]]

        faces_str = [' '.join(x) for x in faces_str]
        return faces_str

    def save_txt(self, file_path, mkdir=True):
        content = self.format_content()

        dirname = os.path.dirname(file_path)
        if not os.path.exists(dirname):
            if mkdir:
                try:
                    os.makedirs(dirname)
                except:
                    pass
            else:
                raise Exception("Dir '{}' not found".format(dirname))
        with open(file_path, 'w') as f:
            f.write('\n'.join(content))

    def __parse_file(self):
        with open(self.label_path, 'r') as f:
            content = f.read()
        self.raw = content.strip('\n')
        self.faces_lines = self.raw.split('\n')  # one line of attributes
        self.faces = []

        for face_num, face in enumerate(self.faces_lines):
            DEBUG_PRINT = False
            t = face.split(' ')

            for i in range(len(t)):
                if 'None' in t[i]: t[i] = None
                try:
                    t[i] = float(t[i])
                except:
                    pass
            if DEBUG_PRINT:
                for attr_num, rs in enumerate(t):
                    print "[{}][{}] {}".format(face_num, attr_num, rs)

            attr_dict = OrderedDict()
            # age 0
            age = t[0]
            attr_dict['age'] = age
            if age is not None and float(age) <= 1: attr_dict['age'] = None
            # blurness 1 2

            blurness = OrderedDict({'threshold': t[1], 'value': t[2]})
            attr_dict['blurness'] = blurness
            # ethnicity
            ethnicity = t[3]
            attr_dict['ethnicity'] = ethnicity
            # left_eye_status [4,9], right_eye_status[10,15]
            left_eye_status = OrderedDict({'dark_glasses': t[4], 'no_glass_eye_close': t[5], 'no_glass_eye_open': t[6],
                                           'normal_glass_eye_close': t[7], 'normal_glass_eye_open': t[8],
                                           'occlusion': t[9]})
            attr_dict['left_eye_status'] = left_eye_status
            right_eye_status = OrderedDict(
                {'dark_glasses': t[10], 'no_glass_eye_close': t[11], 'no_glass_eye_open': t[12],
                 'normal_glass_eye_close': t[13], 'normal_glass_eye_open': t[14], 'occlusion': t[15]})
            attr_dict['right_eye_status'] = right_eye_status
            # facequality 16,17
            facequality = OrderedDict({'threshold': t[16], 'value': t[17]})
            attr_dict['facequality'] = facequality
            # gender 18
            gender = t[18]
            attr_dict['gender'] = gender
            # glass 19
            glass = t[19]
            attr_dict['glass'] = glass
            # headpose 20 21 22
            headpose = OrderedDict({'pitch_angle': t[20], 'roll_angle': t[21], 'yaw_angle': t[22]})
            attr_dict['headpose'] = headpose
            # smile 23 24
            smile = OrderedDict({'threshold': t[23], 'value': t[24]})
            attr_dict['smile'] = smile
            # face_rectangle 25 26 27 28 #hxyw

            face_rectangle = OrderedDict({'height': t[25], 'left': t[26], 'top': t[27], 'width': t[28]})

            face_rectangle_xyxy = np.array([t[26], t[27], t[26] + t[28], t[27] + t[25]])

            attr_dict['face_rectangle'] = face_rectangle
            attr_dict['face_rectangle_xyxy'] = face_rectangle_xyxy

            # face_token 29
            face_token = t[29]
            attr_dict['face_token'] = face_token

            # 83 or 68 or None
            # keypoints 30~195
            if t[30] == None:  # read from new label, and there is no face
                attr_dict['keypoints83'] = None
                attr_dict['keypoints'] = None
            else:
                try:
                    keypoints = np.array(t[30:]).reshape(-1, 2)
                except:
                    pass
                if keypoints.shape[0] == 83:  # read a original file
                    attr_dict['keypoints83'] = keypoints
                    attr_dict['keypoints'] = landmark_reduce(keypoints)
                else:
                    attr_dict['keypoints83'] = None
                    attr_dict['keypoints'] = keypoints
                # elif keypoints.shape[0] == 68:  # read a new label file, and there is a face68
                #     attr_dict['keypoints83'] = None
                #     attr_dict['keypoints'] = keypoints
                # elif keypoints.shape[0] == 19:
                #     attr_dict['keypoints83'] = None
                #     attr_dict['keypoints'] = keypoints
                # elif keypoints.shape[0] == 21:
                #     attr_dict['keypoints83'] = None
                #     attr_dict['keypoints'] = keypoints
                # elif keypoints.shape[0] == 34:
                #     attr_dict['keypoints83'] = None
                #     attr_dict['keypoints'] = keypoints


            # append this face
            self.faces.append(attr_dict)

    def check_arg4version(self, arg4version):
        valid_list = ['_attributes', '_label_washed', '_label_nose', '_label_agr', None]
        assert arg4version in valid_list, \
            '\nFailed to initialize FaceImage Object: Unknown value "{}" for label_suffix arugment.\nAvailable suffixs are:\n{}'.format(
                arg4version,
                ''.join(['\t"{}"\n'.format(asl) for asl in valid_list])
            )
        return arg4version

    '''
        'file_path' supports [image file path] or [label file path]
        By given a [label_suffix], to specify different version of labels.
        Default value of [label_suffix] should be set to the lastest version of label DB.
        Also, a [valid_list] in [check_label_suffix] method should be well maintained.

        Example:
            [image file path] has form:
                <?/?/?>/DATASET_ROOT/imdb2/imdb3001_4000/nm0000012/0eb96830e827853c6f14a641eecebdfac0a6c337.jpg
            [label file path] has form:
                <?/?/?>/DATASET_ROOT/imdb2[label_suffix]/imdb3001_4000/nm0000012/0eb96830e827853c6f14a641eecebdfac0a6c337.txt

        By providing one of paths above, the corresponding one will be found automatically.
        attributes of each face will be parsed into a dict, and 'self.faces' will contains instances of these dicts.

    '''

    def __init__(self, file_path, arg4version=None):

        self.arg4version = self.check_arg4version(arg4version)
        # init: image_path, label_path

        self.__set_lbl_img_path(file_path, arg4version)

        if not os.path.exists(self.image_path):
            original_suffix = self.image_path.split('.')[-1]
            revised_suffix = 'jpg' if original_suffix == 'png' else 'png'
            self.image_path = self.image_path.replace(original_suffix, revised_suffix)
            assert os.path.exists(self.image_path), self.image_path
        assert os.path.exists(self.label_path), self.label_path

        # init: attributes

        self.__parse_file()

        self.opened_image = None


if __name__ == '__main__':
    from mylab.tool import *
    from joblib import Parallel, delayed
    import traceback


    def check_one_fi(l, task_id, task_num):
        precent1 = int(0.01 * task_num)
        if task_id % precent1 == 0:
            print '({: 3d}%) {: 6d}/{: 6d}'.format(task_id // precent1, task_id, task_num)
        FaceImage(l)


    @timeit
    def check_fi_list(file_list):
        len_ = len(file_list)
        parallelizer = Parallel(n_jobs=12)
        taskers = (delayed(check_one_fi)(l, i, len_) for i, l in enumerate(file_list))

        parallelizer(taskers)

        print 'Done'


    '''generate label txt'''
    '''
    db_name = 'imdb2'
    img_list = get_im_list(db_name, clean=True)
    det_list = get_det_result(db_name, clean=True)


    pbar = ProgressBar(len(img_list))
    error_list = []
    error_trace = []
    for im, dl in zip(img_list, det_list):
        try:
            pbar += 1

            f = FaceImage(im, default_lbl_washed=False) #load old label(83)
            f.insert_newbox(dl)

            label_path = f.label_path
            label_path = label_path.split('/')
            label_path[-4] = 'imdb2_label_washed' if 'imdb2' in label_path[-4] else 'imdb_label_washed'
            label_path = '/'.join(label_path)
            dir_path = os.path.dirname(label_path)
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                except:
                    print traceback.format_exc()

            f.save_txt(label_path)


        except:
            error_list.append(im)
            error_trace.append(traceback.format_exc())
            print traceback.format_exc()

    print 'error num:',len(error_list)
    if len(error_list) > 0:
        from mylab.tool import save_dict
        save_dict('./error.npy', {'error_list':error_list, 'error_trace':error_trace})
        print 'saved to ./error.npy'
    '''

    '''check after generating'''
    ls = ['/data5/dataset/MulSrcData/aflw-pifa/trainset/images/6.jpg',
          '/data5/dataset/MulSrcData/aflw-pifa/trainset/images/10.jpg',
          '/data5/dataset/MulSrcData/aflw-pifa/trainset/images/116.jpg',
          '/data5/dataset/MulSrcData/aflw-full/trainset/images/0_image00430_1.jpg',
          '/data5/dataset/MulSrcData/aflw-full/trainset/images/0_image00580_1.jpg',
          '/data5/dataset/MulSrcData/aflw-full/trainset/images/0_image00724_1.jpg',
          '/data5/dataset/MulSrcData/cofw/trainset/images/100.jpg',
          '/data5/dataset/MulSrcData/cofw/trainset/images/1.jpg',
          '/data5/dataset/MulSrcData/cofw/trainset/images/32.jpg',
          '/data5/dataset/MulSrcData/300w-c(re)/trainval/images/153057847_1_s1_r0_x0_y0.jpg',
          '/data5/dataset/MulSrcData/aflw-full/trainset/images/2_image09437_1.jpg',
          '/data5/dataset/MulSrcData/afw/trainval/images/4237203680_2.jpg',
          '/data5/dataset/MulSrcData/helen/trainset/images/2185205964_2.jpg',
          '/data5/dataset/MulSrcData/lfpw/trainset/images/image_0762.png']

    # ls = ['/data5/dataset/MulSrcData/300w-c1(ti)/trainval/images/afw_4237203680.jpg',
    #      '/data5/dataset/MulSrcData/300w-c1(ti)/trainval/images/helen_2185205964_2.jpg',
    #      '/data5/dataset/MulSrcData/300w-c1(ti)/trainval/images/lfpw_image_0762.png']

    for l in ls:
        fi = FaceImage(l)
        if l.find('cofw') != -1 or l.find('aflw-full') != -1:
            fi.vis_test(vis_kp_style=2)
        elif l.find('aflw-pifa') != -1:
            # fi.vis_test(vis_kp_style=1)
            fi.vis_test(vis_kp_style=1, vis_text=1)
        else:
            fi.vis_test(vis_kp_style=1, vis_oribox=0)
        pass


    # '''check after generating'''
    # img_list, version_list = get_mulsrc_img_list(['300w-c1_trainval'])
    #
    # for l, v in zip(img_list, version_list):
    #     fi = FaceImage(l)
    #     fi.vis()

    # img_list, version_list = get_agr_img_list(['imdb', 'imdb2'])
    #
    # for l, v in zip(img_list, version_list):
    #     fi = FaceImage(l, v)
    #     fi.vis()

    # label_list = get_washed_label_list('all')
    # for i in range(len(label_list)):
    #     i = np.random.randint(0,len(label_list))
    #     f = FaceImage(label_list[i])
    #     f.vis()


    # washed_label = get_washed_label_list('all')
    # pbar = ProgressBar(len(washed_label))
    # error_list = []
    # for l in washed_label:
    #     pbar+=1
    #
    #     fi = FaceImage(l)
    #
    #
    # print '\nerr num: {}'.format(len(error_list))
    # for e in error_list:
    #     print e



    ''' vis compare before and after new box inserted '''
    # img_list = get_im_list('imdb2', clean=True)
    # det_list = get_det_result('imdb2', clean=True)
    # for im, de in zip(img_list, det_list):
    #     fi = FaceImage(im, default_lbl_washed=False)
    #     print 'box_num: {}'.format(fi.get_bboxes().shape[0])
    #     fi.vis()
    #
    #     fi.insert_newbox(de)
    #     print 'box_num: {}'.format(fi.get_bboxes().shape[0])
    #     fi.vis()
