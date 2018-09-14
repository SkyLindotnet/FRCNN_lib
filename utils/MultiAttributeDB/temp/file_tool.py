import os
import warnings
import numpy as np

this_dir = os.path.dirname(__file__)
mulsrc_keys = ['lfpw_testset', 'afw_trainval', 'helen_testset', 'lfpw_trainset', 'helen_trainset', 'ibug_trainval']
imdb_keys = ['imdb', 'imdb2']

def png_txt(file_path):
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


def get_suffix(paths):
    return ['.png' if 'lfpw' in p else '.jpg' for p in paths]

def ft_help():
    print 'Avaiable dataset keys are:'
    tmp = []
    tmp.extend(mulsrc_keys)
    tmp.extend(imdb_keys)
    for t in tmp:
        print '\t{}'.format(t)

def read_line_file(fp):
    with open(fp, 'r') as f:
        lines = f.read().strip().split('\n')

    # this commented line is only to prevent the blank line problems in ZSF's result
    # lines = ['0 0 0 0 0' if l == '' else l for l in lines]
    return lines


def get_det_result(type='all', clean=True):
    assert type in ['imdb', 'imdb2', 'all']
    d = 'sfz_det' if not clean else 'washed_det'
    imdb_list1 = '{}/{}/det_results/imdb.txt'.format(this_dir, d)
    imdb_list2 = '{}/{}/det_results/imdb2.txt'.format(this_dir, d)

    if type == 'imdb':
        return read_line_file(imdb_list1)

    if type == 'imdb2':
        return read_line_file(imdb_list2)

    if type == 'all':
        i1 = read_line_file(imdb_list1)
        i2 = read_line_file(imdb_list2)
        i1.extend(i2)

        return i1


def get_im_list(type='all', clean=True):
    warnings.warn("Deprecated usage of 'get_im_list'. Use 'get_original_img_list' instead.")
    assert type in ['imdb', 'imdb2', 'all']
    d = 'sfz_det' if not clean else 'washed_det'

    # sfz_det == original0 imagelist
    # washed_det == washed1 imagelist
    imdb_list1 = '{}/{}/img_list/imdb.txt'.format(this_dir, d)
    imdb_list2 = '{}/{}/img_list/imdb2.txt'.format(this_dir, d)

    if type == 'imdb':
        return read_line_file(imdb_list1)

    if type == 'imdb2':
        return read_line_file(imdb_list2)

    if type == 'all':
        i1 = read_line_file(imdb_list1)
        i2 = read_line_file(imdb_list2)
        i1.extend(i2)

        return i1


def get_washed_label_list(type='all'):
    warnings.warn("Deprecated usage of 'get_washed_label_list'. Use 'get_washed_lbl_list' instead.")
    assert type in ['imdb', 'imdb2', 'all']
    imdb_list1 = '/data5/dataset/imdb_attributes/washed_label_list.txt'
    imdb_list2 = '/data5/dataset/imdb_attributes/washed_label_list2.txt'

    if type == 'imdb':
        return read_line_file(imdb_list1)

    if type == 'imdb2':
        return read_line_file(imdb_list2)

    if type == 'all':
        i1 = read_line_file(imdb_list1)
        i2 = read_line_file(imdb_list2)
        i1.extend(i2)

        return i1


'''-----------------IMDB--DATASET-----------------------'''


def super_get_list(name_path_dict, names):
    if type(names) is str:
        assert names in name_path_dict.keys(), 'Unknown key: {}'.format(names)
        paths = read_line_file(name_path_dict[names])

        return paths

    # else:
    paths = []

    old_names = names
    names = np.unique(names)
    if len(old_names) != len(names):
        print 'Duplicated requests has been removed, from {} to {}'.format(old_names, names)
    for name in names:
        assert name in name_path_dict.keys(), 'Unknown key: {}'.format(name)

    for l in name_path_dict.keys():
        assert os.path.exists(name_path_dict[l]), "'{}':'{}'".format(l, name_path_dict[l])

    for name in names:
        lst = read_line_file(name_path_dict[name])
        paths.extend(lst)

    return paths


'''
[0] Original image and label list
'''


def get_original_img_list(names):
    paths, arg4version = get_original_lbl_list(names)
    for l in paths:
        t = l.replace('.txt', '.jpg').split('/')
        t[-4] = t[-4].replace('_attributes', '')
        l = '/'.join(t)
    return paths, arg4version


def get_original_lbl_list(names):
    lst1 = '/data5/dataset/imdb_attributes/label_lists/original0/imdb_label_list.txt'
    lst2 = '/data5/dataset/imdb_attributes/label_lists/original0/imdb2_label_list.txt'

    d = {'imdb': lst1, 'imdb2': lst2}
    paths = super_get_list(d, names)

    arg4version = ['_attributes' for _ in range(len(paths))]

    return paths, arg4version


'''
[1] Washed image and label list
    @remove ref.jpg
    @remove face++ 0 dets images
    @remove zsf 0 dets images
    @insert zsf box into face++ box
    @compute 83 point to 68 (with bad results on nose)
'''


def get_washed_img_list(names):
    paths, arg4version = get_washed_lbl_list(names)
    paths = [l.replace('.txt', '.jpg').replace('_label_washed', '') for l in paths]
    return paths, arg4version


def get_washed_lbl_list(names):
    lst1 = '/data5/dataset/imdb_attributes/label_lists/washed1/washed_label_list.txt'
    lst2 = '/data5/dataset/imdb_attributes/label_lists/washed1/washed_label_list2.txt'

    d = {'imdb': lst1, 'imdb2': lst2}
    paths = super_get_list(d, names)

    arg4version = ['_label_washed' for _ in range(len(paths))]
    return paths, arg4version


'''
[2] Nose image and label list
    @use yudi's matlab get a better nose results
    @replace bad noses with these refined noses
'''


def get_nose_img_list(names):
    paths, arg4version = get_nose_lbl_list(names)
    paths = [l.replace('.txt', '.jpg').replace('_label_nose', '') for l in paths]

    return paths, arg4version


def get_nose_lbl_list(names):
    lst1 = '/data5/dataset/imdb_attributes/label_lists/nose2/imdb_label_nose_list.txt'
    lst2 = '/data5/dataset/imdb_attributes/label_lists/nose2/imdb2_label_nose_list.txt'

    d = {'imdb': lst1, 'imdb2': lst2}
    paths = super_get_list(d, names)
    arg4version = ['_label_nose' for _ in range(len(paths))]
    return paths, arg4version


'''
[3] AGR image and label list
    @use tanzichang's model to get refined result of age, gender, race
    @replace face++'s AGR result if there are no None in [age, gender, race] for a face

'''


def get_agr_img_list(names):
    paths, arg4version = get_agr_lbl_list(names)
    paths = [l.replace('.txt', '.jpg').replace('_label_agr', '') for l in paths]
    return paths, arg4version


def get_agr_lbl_list(names):
    lst1 = '/data5/dataset/imdb_attributes/label_lists/agr3/imdb_label_agr_list.txt'
    lst2 = '/data5/dataset/imdb_attributes/label_lists/agr3/imdb2_label_agr_list.txt'

    d = {'imdb': lst1, 'imdb2': lst2}
    paths = super_get_list(d, names)
    arg4version = ['_label_agr' for _ in range(len(paths))]
    return paths, arg4version


'''-----------------MULTI-SOURCE-DATASET-----------------------'''

'''
[0] original
version code: "labels"
'''
def get_mulsrc_img_list(names):
    paths, arg4version = get_mulsrc_lbl_list(names)
    version = 'labels'

    suffixs = get_suffix(paths)
    paths = [l.replace('.txt', sf).replace(version, 'images') for l, sf in zip(paths, suffixs)]
    return paths, arg4version


def get_mulsrc_lbl_list(names):
    afw_trainval = '/data5/dataset/MulSrcData/label_lists/original0/afw_trainval_lbl_list.txt'
    helen_testset = '/data5/dataset/MulSrcData/label_lists/original0/helen_testset_lbl_list.txt'
    helen_trainset = '/data5/dataset/MulSrcData/label_lists/original0/helen_trainset_lbl_list.txt'
    ibug_trainval = '/data5/dataset/MulSrcData/label_lists/original0/ibug_trainval_lbl_list.txt'
    lfpw_testset = '/data5/dataset/MulSrcData/label_lists/original0/lfpw_testset_lbl_list.txt'
    lfpw_trainset = '/data5/dataset/MulSrcData/label_lists/original0/lfpw_trainset_lbl_list.txt'

    d = {}
    d['afw_trainval'] = afw_trainval
    d['helen_testset'] = helen_testset
    d['helen_trainset'] = helen_trainset
    d['ibug_trainval'] = ibug_trainval
    d['lfpw_testset'] = lfpw_testset
    d['lfpw_trainset'] = lfpw_trainset
    paths = super_get_list(d, names)
    arg4version = ['labels' for _ in range(len(paths))]
    return paths, arg4version


'''
[latest] Maintain a latest image and label list
'''




def get_latest_img_list(names, shuffle=False):
    names = [names] if type(names) is str else names
    mulsrc_names = []
    imdb_names = []
    for n in names:
        if n in mulsrc_keys:
            mulsrc_names.append(n)
        elif n in imdb_keys:
            imdb_names.append(n)
        else:
            raise Exception("\nUnknown key: '{}'.\nAvailable keys are:\n\t{}\n\t{}".format(n, '\n\t'.join(mulsrc_keys),
                                                                                           '\n\t'.join(imdb_keys)))
    assert len(mulsrc_names) + len(imdb_names) > 0

    P = []  # all paths

    if len(mulsrc_names) != 0:
        p, a = get_mulsrc_img_list(mulsrc_names)
        P.extend(p)

    if len(imdb_names) != 0:
        p, a = get_agr_img_list(imdb_names)
        P.extend(p)

    if shuffle: np.random.shuffle(P)

    return P

def get_latest_lbl_list(names, shuffle=False):
    names = [names] if type(names) is str else names
    mulsrc_names = []
    imdb_names = []
    for n in names:
        if n in mulsrc_keys:
            mulsrc_names.append(n)
        elif n in imdb_keys:
            imdb_names.append(n)
        else:
            raise Exception("\nUnknown key: '{}'.\nAvailable keys are:\n\t{}\n\t{}".format(n, '\n\t'.join(mulsrc_keys),
                                                                                           '\n\t'.join(imdb_keys)))
    assert len(mulsrc_names) + len(imdb_names) > 0

    P = []  # all paths
    if len(mulsrc_names) != 0:
        p, a = get_mulsrc_lbl_list(mulsrc_names)
        P.extend(p)

    if len(imdb_names) != 0:
        p, a = get_agr_lbl_list(imdb_names)
        P.extend(p)

    if shuffle: np.random.shuffle(P)


    return P


if __name__ == '__main__':
    from mylab.tool import *
    from FaceImage import FaceImage

    N = []
    N.extend(mulsrc_keys)
    N.extend(imdb_keys)

    imgl = get_latest_img_list(N)[0:]
    lbll = get_latest_lbl_list(N)[0:]

    pbar = ProgressBar(len(imgl))

    for i in imgl:

        # print a
        fi = FaceImage(i)
        pbar += 1

    pbar = ProgressBar(len(lbll))
    for i in lbll:
        fi = FaceImage(i)
        pbar += 1
