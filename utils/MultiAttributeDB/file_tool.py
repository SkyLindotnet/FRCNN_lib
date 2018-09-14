import os
import warnings
import numpy as np

this_dir = os.path.dirname(__file__)

mulsrc_keys = ['lfpw_testset', 'afw_trainval', 'helen_testset',
               'lfpw_trainset', 'helen_trainset', 'ibug_trainval',
               '300w-c1_trainval', '300w-c1(ti)_trainval', '300w-c(re)_trainval',
               '300w(re)_trainval', 'aflw-pifa_trainset', 'cofw_trainset',
               'aflw-full_trainset']
imdb_keys = ['imdb', 'imdb2']
wider_keys = ['wider_train', 'wider_val']
morph_keys = ['morph_s1_train', 'morph_s1_test', 'morph_s2_train', 'morph_s2_test']
facepp_keys = ['facepp_train', 'facepp_val', 'facepp_test']
frgc_keys = ['frgc_trainval']

keys_list = [mulsrc_keys, imdb_keys, wider_keys, morph_keys, facepp_keys]


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


def ft_help(do_print=True):
    print 'Avaiable dataset keys are:'

    tmp = []
    for kl in keys_list:
        tmp.extend(kl)
    s = ''
    for t in tmp:
        s += '\t{}\n'.format(t)
    if do_print: print s
    return s


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


def get_suffix(paths):
    return ['.png' if 'lfpw' in p else '.jpg' for p in paths]


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
    threeHus_trainval = '/data5/dataset/MulSrcData/label_lists/original0/300w_trainval_lbl_list.txt'
    threeHus_re_trainval = '/data5/dataset/MulSrcData/label_lists/original0/300w(re)_trainval_lbl_list.txt'
    threeHus_c_trainval = '/data5/dataset/MulSrcData/label_lists/original0/300w-c_trainval_lbl_list.txt'
    threeHus_c_re_trainval = '/data5/dataset/MulSrcData/label_lists/original0/300w-c(re)_trainval_lbl_list.txt'
    threeHus_c1_trainval = '/data5/dataset/MulSrcData/label_lists/original0/300w-c1_trainval_lbl_list.txt'
    threeHus_c1_ti_trainval = '/data5/dataset/MulSrcData/label_lists/original0/300w-c1(ti)_trainval_lbl_list.txt'
    aflw_full_trainset = '/data5/dataset/MulSrcData/label_lists/original0/aflw-full_trainset_lbl_list.txt'
    aflw_full_testset = '/data5/dataset/MulSrcData/label_lists/original0/aflw-full_testset_lbl_list.txt'
    aflw_pifa_trainset = '/data5/dataset/MulSrcData/label_lists/original0/aflw-pifa_trainset_lbl_list.txt'
    aflw_pifa_testset = '/data5/dataset/MulSrcData/label_lists/original0/aflw-pifa_testset_lbl_list.txt'
    cofw_trainset = '/data5/dataset/MulSrcData/label_lists/original0/cofw_trainset_lbl_list.txt'
    cofw_testset = '/data5/dataset/MulSrcData/label_lists/original0/cofw_testset_lbl_list.txt'

    d = {}
    d['afw_trainval'] = afw_trainval
    d['helen_testset'] = helen_testset
    d['helen_trainset'] = helen_trainset
    d['ibug_trainval'] = ibug_trainval
    d['lfpw_testset'] = lfpw_testset
    d['lfpw_trainset'] = lfpw_trainset
    d['300w_trainval'] = threeHus_trainval
    d['300w(re)_trainval'] = threeHus_re_trainval
    d['300w-c_trainval'] = threeHus_c_trainval
    d['300w-c(re)_trainval'] = threeHus_c_re_trainval
    d['300w-c1_trainval'] = threeHus_c1_trainval
    d['300w-c1(ti)_trainval'] = threeHus_c1_ti_trainval
    d['aflw-full_testset'] = aflw_full_testset
    d['aflw-full_trainset'] = aflw_full_trainset
    d['aflw-pifa_testset'] = aflw_pifa_testset
    d['aflw-pifa_trainset'] = aflw_pifa_trainset
    d['cofw_testset'] = cofw_testset
    d['cofw_trainset'] = cofw_trainset
    paths = super_get_list(d, names)
    arg4version = ['labels' for _ in range(len(paths))]
    return paths, arg4version


'''-----------------WIDER-FACE-DATASET-----------------------'''

'''
[0] original
version code: "Labels"
'''


def get_wider_img_list(names):
    paths, arg4version = get_wider_lbl_list(names)
    version = 'Labels'
    paths = [p.replace(version, 'Images').replace('.txt', '.jpg') for p in paths]
    return paths, arg4version


def get_wider_lbl_list(names):
    wider_train = '/data5/dataset/MulSrcData/voc_wider/data/Labels/train_lbl_list.txt'
    wider_val = '/data5/dataset/MulSrcData/voc_wider/data/Labels/val_lbl_list.txt'

    d = {}
    d['wider_train'] = wider_train
    d['wider_val'] = wider_val
    paths = super_get_list(d, names)
    arg4version = ['Labels' for _ in range(len(paths))]
    return paths, arg4version


'''-----------------MORPH-DATASET-----------------------'''


def get_morph_img_list(names):
    paths, arg4version = get_morph_lbl_list(names)
    version = 'labels'
    paths = [p.replace(version, 'images').replace('.txt', '.jpg') for p in paths]
    return paths, arg4version


def get_morph_lbl_list(names):
    s1_train = '/home/ylxie/MulSrcData/label_lists/original0/morph_S1_Train_lbl_list.txt'
    s1_test = '/home/ylxie/MulSrcData/label_lists/original0/morph_S1_Test_lbl_list.txt'
    s2_train = '/home/ylxie/MulSrcData/label_lists/original0/morph_S2_Train_lbl_list.txt'
    s2_test = '/home/ylxie/MulSrcData/label_lists/original0/morph_S2_Test_lbl_list.txt'

    d = {}
    d['morph_s1_train'] = s1_train
    d['morph_s1_test'] = s1_test
    d['morph_s2_train'] = s2_train
    d['morph_s2_test'] = s2_test
    paths = super_get_list(d, names)
    arg4version = ['labels' for _ in range(len(paths))]
    return paths, arg4version


'''-----------------FACE.PP-DATASET-----------------------'''


def get_facepp_img_list(names):
    paths, arg4version = get_facepp_lbl_list(names)
    version = 'labels'
    paths = [p.replace(version, 'images').replace('.txt', '.jpg') for p in paths]
    return paths, arg4version


def get_facepp_lbl_list(names):
    train = '/data5/dataset/MulSrcData/label_lists/original0/facepp_train_lbl_list.txt'
    val = '/data5/dataset/MulSrcData/label_lists/original0/facepp_val_lbl_list.txt'
    test = '/data5/dataset/MulSrcData/label_lists/original0/facepp_test_lbl_list.txt'

    d = {}
    d['facepp_train'] = train
    d['facepp_val'] = val
    d['facepp_test'] = test
    paths = super_get_list(d, names)
    arg4version = ['labels' for _ in range(len(paths))]
    return paths, arg4version


'''-----------------FRGC-DATASET-----------------------'''


def get_frgc_img_list(names):
    paths, arg4version = get_frgc_lbl_list(names)
    version = 'labels'
    paths = [p.replace(version, 'images').replace('.txt', '.jpg') for p in paths]
    return paths, arg4version


def get_frgc_lbl_list(names):
    trainval = '/data5/dataset/MulSrcData/label_lists/original0/frgc_trainval_lbl_list.txt'
    d = {}
    d['frgc_trainval'] = trainval

    paths = super_get_list(d, names)
    arg4version = ['labels' for _ in range(len(paths))]
    return paths, arg4version


'''
[latest] Maintain a method which returns the latest version of image and label list
'''

# edit here to append a new dataset
keys_getter_map = {}
keys_getter_map[tuple(mulsrc_keys)] = [get_mulsrc_img_list, get_mulsrc_lbl_list]
keys_getter_map[tuple(imdb_keys)] = [get_agr_img_list, get_agr_lbl_list]
keys_getter_map[tuple(wider_keys)] = [get_wider_img_list, get_wider_lbl_list]
keys_getter_map[tuple(morph_keys)] = [get_morph_img_list, get_morph_lbl_list]
keys_getter_map[tuple(facepp_keys)] = [get_facepp_img_list, get_facepp_lbl_list]
keys_getter_map[tuple(frgc_keys)] = [get_frgc_img_list, get_frgc_lbl_list]


def distribute_names(names, getter_type):
    assert getter_type in ['img', 'lbl'], "Unknown getter_type: {}. Available types are 'img' and 'lbl'"
    names = [names] if type(names) == str else names
    keys_of_map = keys_getter_map.keys()
    distributed_names = [[] for _ in range(len(keys_of_map))]
    for n in names:
        found_match = False
        for i, k in enumerate(keys_of_map):
            if n in k:
                distributed_names[i].append(n)
                found_match = True
                break
        if not found_match:
            raise Exception("\nUnknown key: '{}'.\n{}".format(n, ft_help(do_print=False)))

    non_emtpy_distributed_names = []
    functions = []

    for i, dk in enumerate(distributed_names):
        if len(dk) == 0: continue

        non_emtpy_distributed_names.append(dk)
        img_lbl_funcs = keys_getter_map[keys_of_map[i]]
        functions.append(img_lbl_funcs[0] if getter_type == 'img' else img_lbl_funcs[1])

    return non_emtpy_distributed_names, functions


def get_latest_img_list(names, shuffle=False):
    distributed_names, functions = distribute_names(names, 'img')
    assert np.sum([len(x) for x in distributed_names]) > 0, 'No tasks assigned here.'

    P = []  # all paths
    for names, function in zip(distributed_names, functions):
        p, a = function(names)
        P.extend(p)
    if shuffle: np.random.shuffle(P)
    return P


def get_latest_lbl_list(names, shuffle=False):
    distributed_names, functions = distribute_names(names, 'lbl')
    assert np.sum([len(x) for x in distributed_names]) > 0, 'No tasks assigned here.'

    P = []  # all paths
    for names, function in zip(distributed_names, functions):
        p, a = function(names)
        P.extend(p)
    if shuffle: np.random.shuffle(P)
    return P


if __name__ == '__main__':
    from mylab.tool import *
    from FaceImage import FaceImage

    N = ['frgc_trainval']
    # N.extend(facepp_keys)
    # N.extend(imdb_keys)
    # for k in keys_list:
    #     N.extend(k)
    #
    imgl = get_latest_img_list(N, shuffle=True)[0:]
    lbll = get_latest_lbl_list(N, shuffle=True)[0:]

    pbar = ProgressBar(len(imgl))
    # a = 0
    for i in imgl:
        #   print a; a+=1
        fi = FaceImage(i)

        pbar += 1

    pbar = ProgressBar(len(lbll))
    for i in lbll:
        fi = FaceImage(i)

        pbar += 1
