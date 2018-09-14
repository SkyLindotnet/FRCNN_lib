import numpy as np
import os, sys
from collections import OrderedDict
# from mylab.draw import *
from pylab import *

class FaceImage():
    def __get_lbl_img_path(self, file_path):
        if file_path.endswith('.jpg'):
            parts = file_path.split('/')
            if parts[-2] == 'ref':
                parts[-5] += '_attributes'
            else:
                parts[-4] += '_attributes'
            parts[-1] = parts[-1].replace('.jpg', '.txt')
            return '/'.join(parts)

        if file_path.endswith('.txt'):
            parts = file_path.split('/')
            parts[-4] = parts[-4].replace('_attributes', '')
            parts[-1] = parts[-1].replace('.txt', '.jpg')
            return '/'.join(parts)

    def get_opened_image(self):
        if self.opened_image is None:
            self.opened_image = imread(self.image_path)
        return self.opened_image

    def get_bboxes(self, order='whxy'):
        ava_orders = ['whxy','xywh','xyxy']
        if order not in ava_orders:
            raise Exception('Bad order: {}, available orders are [{}]'.format(order, ', '.join(ava_orders)))

        ret = np.zeros(shape=(len(self.faces), 4))
        for i,f in enumerate(self.faces):
            if order == 'whxy':
                ret[i,:] = f['face_rectangle_whxy']
            elif order == 'xywh':
                ret[i,:] = f['face_rectangle_xywh']
            elif order == 'xyxy':
                ret[i,:] = f['face_rectangle_xyxy']
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

    def get_keypoints(self):
        ret = np.zeros(shape=(len(self.faces), 83,2))
        for i, f in enumerate(self.faces):
            ret[i,:,:] = f['keypoints']
        return ret

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
            # blurness 1 2
            blurness = {'threshold': t[1], 'value': t[2]}
            attr_dict['blurness'] = blurness
            # ethnicity
            ethnicity = t[3]
            attr_dict['ethnicity'] = ethnicity
            # left_eye_status [4,9], right_eye_status[10,15]
            left_eye_status = {'dark_glasses': t[4], 'no_glass_eye_close': t[5], 'no_glass_eye_open': t[6],
                               'normal_glass_eye_close': t[7], 'normal_glass_eye_open': t[8], 'occlusion': t[9]}
            attr_dict['left_eye_status'] = left_eye_status
            right_eye_status = {'dark_glasses': t[10], 'no_glass_eye_close': t[11], 'no_glass_eye_open': t[12],
                                'normal_glass_eye_close': t[13], 'normal_glass_eye_open': t[14], 'occlusion': t[15]}
            attr_dict['right_eye_status'] = right_eye_status
            # facequality 16,17
            facequality = {'threshold': t[16], 'value': t[17]}
            attr_dict['facequality'] = facequality
            # gender 18
            gender = t[18]
            attr_dict['gender'] = gender
            # glass 19
            glass = t[19]
            attr_dict['glass'] = glass
            # headpose 20 21 22
            headpose = {'pitch_angle': t[20], 'roll_angle': t[21], 'yaw_angle': t[22]}
            attr_dict['headpose'] = headpose
            # smile 23 24
            smile = {'threshold': t[23], 'value': t[24]}
            attr_dict['smile'] = smile
            # face_rectangle 25 26 27 28
            face_rectangle = {'height': t[25], 'left': t[26], 'top': t[27], 'width': t[28]}
            face_rectangle_whxy = np.array([t[28], t[25], t[26], t[27]]).reshape(1, 4)
            face_rectangle_xywh = np.array([t[26], t[27], t[28], t[25]]).reshape(1, 4)
            face_rectangle_xyxy = np.array([t[26], t[27], t[26] + t[28], t[27] + t[25]])

            attr_dict['face_rectangle'] = face_rectangle
            attr_dict['face_rectangle_whxy'] = face_rectangle_whxy
            attr_dict['face_rectangle_xywh'] = face_rectangle_xywh
            attr_dict['face_rectangle_xyxy'] = face_rectangle_xyxy

            # face_token 29
            face_token = t[29]
            attr_dict['face_token'] = face_token
            # keypoints 30~195
            keypoints = np.array(t[30:]).reshape(83, 2)
            attr_dict['keypoints'] = keypoints
            self.faces.append(attr_dict)

    '''
        'file_path' supports [image file path] or [label file path]
        Assuming
            [image file path] has form:
                <?/?/?>/DATASET_ROOT/imdb/imdb3001_4000/nm0000012/0eb96830e827853c6f14a641eecebdfac0a6c337.jpg
            [label file path] has form:
                <?/?/?>/DATASET_ROOT/imdb_attributes/imdb3001_4000/nm0000012/0eb96830e827853c6f14a641eecebdfac0a6c337.txt

        By providing one, another will be found automatically.
        attributes of each face  will be parsed into a dict, and 'self.faces' will contains instances of these dicts.

    '''
    def __init__(self, file_path):
        # init: image_path, label_path
        if file_path.endswith('.jpg'):
            self.image_path = file_path
            self.label_path = self.__get_lbl_img_path(file_path)
        elif file_path.endswith('.txt'):
            self.image_path = self.__get_lbl_img_path(file_path)
            self.label_path = file_path
        assert os.path.exists(self.image_path), self.image_path
        assert os.path.exists(self.label_path), self.label_path

        # init: attributes
        self.__parse_file()
        self.opened_image = None

