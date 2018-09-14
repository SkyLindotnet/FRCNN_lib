import numpy as np
from pylab import *
import matplotlib.patches as patches
import numpy.random as npr
from PIL import Image
import os,sys
from .io import *
# get path_file by: cd ~ && find $FDDB_WORKSPACE -name "*.jpg"  > FDDB_all.txt
path_file = '/home/rick/Space/work/FDDB/data/Annotations/FDDB_all.txt'
with open(path_file, 'r') as f:
    FDDB_all = f.readlines()

def get_fddb_fcn(s):
    '''

    :param s:
    Supporting the following formats, where '?' means doesn't matter:
        TYPE1[underscore] : ??/....??/??/??/2002_08_02_big_img_198.jpg
        TYPE2[pathformat] : ??/??/....??/??/??/2002/08/02/big/img_198.jpg
        TYPE3 : An integer index in range [0,2845)
    :return: fg, bg, out(i.e. argmaxed value)
    '''

    if type(s) == int:
        s = get_fddb_img(num=s,imgread=False,is_id=True)

    fcn_out_dir = '/home/rick/Space/work/FDDB/FCN_OUT'
    s = s.split('.')[0].replace('\n','').replace(' ','')

    if len(s.split('_')) > 3: s = s.split('/')[-1]
    else: s = '_'.join(s.split('/')[-5:])

    target = os.path.join(fcn_out_dir,s+'.npy')
    assert os.path.exists(target), "[!] {} not found".format(target.replace('.npy',' .npy'))
    D = load_dict(target)
    return D['fg'], D['bg'], D['out'], 

def get_fddb_list():
    '''

    :return: full path list of fddb images
    '''
    return [ll.replace("\n",'') for ll in FDDB_all]

def get_fddb_img(num=1, imgread=True,usePIL=True,is_id=False):
    '''

    :param num:
        Use as number of images required:
            if give 1: return a single image
            if > 1 : return a list of image
        Use as id of images:
            return the num-th image
    :param imgread:
        Read the image or just return the image path
        if imgread: return an opened image
        if not imgread: return a image path
    :param usePIL:
        The way to open image:
            if usePIL: read image using PIL.Image.open
            if not usePIL: read image using pylab.imread
    :param is_id:
        Treating the 1st variable(num) as ID: in range [0,2845)
    :return:
        Either opened image(s) or just image path(s)

    '''
    img_path = get_fddb_list()
    num_img = len(img_path)
    if is_id: #use num as ID instead of number of requests
        if num >= num_img or num < 0:
            return None
        img = img_path[num]
        if imgread:
            print "Image: {}".format(img)
            if usePIL: img = Image.open(img_path[num])
            else: img = imread(img_path[num])
        return img

    assert num<=num_img, "too many required images"
    idx = npr.choice(num_img,num, replace=False)
    img_path = np.array(img_path)
    imgs = img_path[idx]

    if imgread:
        for i in imgs: print "Image: {}".format(i)
        if usePIL: imgs = [Image.open(x) for x in imgs]
        else: imgs = [imread(x) for x in imgs]

    if(num == 1):
        return imgs[0]
    else:
        return imgs
