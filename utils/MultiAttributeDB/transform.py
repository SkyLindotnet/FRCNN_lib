# import _init_paths
import numpy as np
import os, sys, shutil
from mylab.tool import mk_dir
from FaceImage import FaceImage
import scipy
import h5py
import pickle
from mylab.draw import *

support_types = ['afw', 'lfpw', 'helen']
pts_savedir = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB'
savedir = '/data5/dataset/MulSrcData/'
label_savedir = '/data5/dataset/MulSrcData/label_lists/original0/'
aflw_image_dir = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/AFLW/aflw/data/flickr'
aflw_full_mat = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/AFLW/aflw/data/AFLWinfo_release.mat'
aflw_pifa_mat = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/AFLW/aflw-pifa/BoundingBox.mat',
                  '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/AFLW/aflw-pifa/Landmark.mat']
aflw_pifa_image_dir = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/AFLW/aflw-pifa/TestImages',
                  '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/AFLW/aflw-pifa/TrainImages']

cofw_mat = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/COFW/COFW_test_color.mat',
            '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/COFW/COFW_train_color.mat']

# assume image and pts files are in the same dir
def mk_img_lb_dir(db_name, type_name):
    img_dir = os.path.join(savedir, db_name, type_name, 'images')
    lb_dir = os.path.join(savedir, db_name, type_name, 'labels')
    mk_dir(img_dir)
    mk_dir(lb_dir)
    return img_dir, lb_dir


# transform pts into faceimage format
def transform_pts(image_dir_path, db_name, type_name='trainval', box_c=0):
    import menpo.io as mio
    from menpo.visualize import print_progress
    img_dir, lb_dir = mk_img_lb_dir(db_name, type_name)
    for img in print_progress(mio.import_images(image_dir_path, verbose=True)):
        img_suffix = img.path.suffix
        lb_suffix = '.txt'
        image_name = img.path.name.split('.')[0]
        img_path = os.path.join(img_dir, image_name + img_suffix)
        lb_path = os.path.join(lb_dir, image_name + lb_suffix)
        # save image
        mio.export_image(img, img_path, overwrite=True)
        # save label
        lb = ['None'] * (30+img.landmarks['PTS'].n_points*2)
        kps = [str(kp) for kp in img.landmarks['PTS'].as_vector() + 1]
        if box_c:
            pickle_path = img.path._str.split('.')[0] + '.pkl'
            with open(pickle_path, 'rb') as f:
                box = pickle.load(f)['box']
                x = str(box[0])
                y = str(box[1])
                w = str(box[2]-box[0])
                h = str(box[3]-box[1])
                box_hxyw = [h] + [x] + [y] + [w]
        else:
            xy = img.landmarks['PTS'].bounds()[0][::-1] + 1
            zk = img.landmarks['PTS'].bounds()[1][::-1] + 1
            x = str(xy[0])
            y = str(xy[1])
            w = str(zk[0] - xy[0])
            h = str(zk[1] - xy[1])
            box_hxyw = [h] + [x] + [y] + [w]
        kps_t = np.array(kps[0::2])
        kps[0::2] = kps[1::2]
        kps[1::2] = kps_t
        lb[30:] = kps
        lb[25:29] = box_hxyw
        with open(lb_path, 'w') as f:
            f.write(' '.join(lb))

        fi = FaceImage(img_path)
        # fi.vis()
    pass


def transform_pts_temp_v1(image_dir_path, db_name, type_name='trainval'):
    import menpo.io as mio
    from menpo.visualize import print_progress
    img_dir, lb_dir = mk_img_lb_dir(db_name, type_name)
    for img in print_progress(mio.import_images(image_dir_path, verbose=True)):
        img_suffix = img.path.suffix
        lb_suffix = '.txt'
        image_name = img.path.name.split('.')[0]

        dataType = filter(lambda x: x in image_name, support_types)[0]
        if dataType == 'afw':
            image_name_list = image_name.split('_')
            image_name_list.pop(2)  # remove index
            image_name = '_'.join(image_name_list)
        img_path = os.path.join(img_dir, image_name + img_suffix)
        lb_path = os.path.join(lb_dir, image_name + lb_suffix)

        # save image
        mio.export_image(img, img_path, overwrite=True)
        # save label
        lb = ['None'] * 166
        kps = [str(kp) for kp in img.landmarks['PTS'].as_vector()]
        xy = img.landmarks['PTS'].bounds()[0][::-1]
        zk = img.landmarks['PTS'].bounds()[1][::-1]
        x = str(xy[0])
        y = str(xy[1])
        w = str(zk[0] - xy[0])
        h = str(zk[1] - xy[1])
        box_hxyw = [h] + [x] + [y] + [w]
        kps_t = np.array(kps[0::2])
        kps[0::2] = kps[1::2]
        kps[1::2] = kps_t
        lb[30:] = kps
        lb[25:29] = box_hxyw
        with open(lb_path, 'a') as f:
            f.write(' '.join(lb) + '\n')

        fi = FaceImage(img_path)
        # fi.vis()
    pass


def transform_pts_temp_v2(image_dir_path, db_name, type_name='trainval'):
    import menpo.io as mio
    from menpo.visualize import print_progress
    img_dir, lb_dir = mk_img_lb_dir(db_name, type_name)
    impaths = filter(lambda x: 'pts' not in x, os.listdir(image_dir_path))
    for imgpath in print_progress(impaths):
        imgpath = os.path.join(image_dir_path, imgpath)
        newpath = transform_impath(imgpath)
        img = mio.import_image(newpath)

        img_suffix = img.path.suffix
        lb_suffix = '.txt'
        image_name = img.path.name.split('.')[0]
        image_type = img.path._str.split('/')[11]
        image_name = '%s_%s' % (image_type, image_name)

        dataType = filter(lambda x: x == image_type, support_types)[0]
        if dataType == 'afw':
            image_name_list = image_name.split('_')
            image_name_list.pop(2)  # remove index
            image_name = '_'.join(image_name_list)
        img_path = os.path.join(img_dir, image_name + img_suffix)
        lb_path = os.path.join(lb_dir, image_name + lb_suffix)

        # save image
        mio.export_image(img, img_path, overwrite=True)
        # save label
        lb = ['None'] * 166
        kps = [str(kp) for kp in (img.landmarks['PTS'].as_vector() + 1)]
        xy = img.landmarks['PTS'].bounds()[0][::-1]
        zk = img.landmarks['PTS'].bounds()[1][::-1]
        x = str(xy[0])
        y = str(xy[1])
        w = str(zk[0] - xy[0])
        h = str(zk[1] - xy[1])
        box_hxyw = [h] + [x] + [y] + [w]
        kps_t = np.array(kps[0::2])
        kps[0::2] = kps[1::2]
        kps[1::2] = kps_t
        lb[30:] = kps
        lb[25:29] = box_hxyw
        with open(lb_path, 'a') as f:
            f.write(' '.join(lb) + '\n')

        fi = FaceImage(img_path)
        # fi.vis()
    pass


def transform_impath(path):
    rootpath = ('/').join(path.split('/')[:6])
    path = ('/').join(path.split('/')[6:])
    import os
    type_str = path.split('/')[5]
    suffix = path.split('/')[-1].split('.')[-1]
    imname = path.split('/')[-1].split('.')[0]
    if type_str == '300w-c1':
        im_type = imname.split('_')[0]
        imdirlist = path.split('/')
        imdirlist[5] = im_type
        imdir = '/'.join(imdirlist[:-1])
        imname = imname.split('_')[1] + '_' + imname.split('_')[2]
        if im_type == 'afw':
            impath = os.path.join(imdir,
                                  '%s.%s' % (imname, suffix))
        else:
            imdirlist = imdir.split('/')
            imdir = '/'.join(imdirlist + ['trainset'])
            impath = os.path.join(imdir,
                                  '%s.%s' % (imname, suffix))
    if not os.path.exists(os.path.join(rootpath, impath.strip())):
        exit(1)
    return os.path.join(rootpath, impath.strip())


def transform_AFLW_full_to_pts(db_name='aflw-full'):
    trainset_dir = os.path.join(pts_savedir, db_name, 'trainset')
    testset_dir = os.path.join(pts_savedir, db_name, 'testset')
    testset_frontal_dir = os.path.join(pts_savedir, db_name, 'testset-frontal')
    mk_dir(trainset_dir)
    mk_dir(testset_dir)
    mk_dir(testset_frontal_dir)

    mat = scipy.io.loadmat(aflw_full_mat)
    nameList = mat['nameList']
    mask_new = mat['mask_new']
    bboxes = mat['bbox']
    kps = mat['data']
    ra = mat['ra']
    train_ra = ra[0, :20000] - 1
    test_ra = ra[0, 20000:] - 1
    for sample_ra, save_dir, im_sum in zip([train_ra, test_ra, test_ra],
                                           [trainset_dir, testset_dir, testset_frontal_dir],
                                           [20000, 4386, 1165]):
        reported_im_names = []
        for im_name, im_kps, im_mask, im_boxes in zip(nameList[sample_ra], kps[sample_ra], mask_new[sample_ra], bboxes[sample_ra]):
            if 'frontal' in save_dir and sum(im_mask) != 19:
                continue
            # save image
            im_name = im_name[0][0].split('.')[0]
            num = reported_im_names.count(im_name) + 1
            re_im_name = '_'.join(im_name.split('/')) + '_%d' % num
            im_path = os.path.join(save_dir, re_im_name + '.jpg')
            ori_im_path = os.path.join(aflw_image_dir, im_name + '.jpg')
            pts_path = im_path.replace('jpg', 'pts')
            pickle_path = im_path.replace('jpg', 'pkl')
            if not os.path.exists(ori_im_path):
                im_path = os.path.join(save_dir, re_im_name + '.png')
                ori_im_path = os.path.join(aflw_image_dir, im_name + '.png')
                pts_path = im_path.replace('png', 'pts')
                pickle_path = im_path.replace('png', 'pkl')
            if os.path.exists(im_path) or os.path.exists(pts_path):
                print im_path
            shutil.copy(ori_im_path, im_path)

            # visual_kp_debug(im_path, im_kps)

            # save pts
            with open(pts_path, 'w') as f:
                f.write('version: 1\n')
                f.write('n_points:  19\n')
                f.write('{\n')
                im_kps = im_kps.reshape([2, -1]).transpose()
                for im_kp in im_kps:
                    f.write('%f %f' % (im_kp[0], im_kp[1]) + '\n')
                f.write('}')

            reported_im_names.append(im_name)
            print 'reported num: %d/%d' % (len(reported_im_names), im_sum)

            # save other information
            with open(pickle_path, 'wb') as f:
                im_boxes = im_boxes.reshape([2, -1]).transpose().ravel()
                info = {'box': im_boxes}
                pickle.dump(info, f)
            # debug
            # with open(pickle_path, 'rb') as f:
            #     newinfo = pickle.load(f)
            #     pass


def transform_AFLW_pifa_to_pts(db_name='aflw-pifa'):
    trainset_dir = os.path.join(pts_savedir, db_name, 'trainset')
    testset_dir = os.path.join(pts_savedir, db_name, 'testset')
    mk_dir(trainset_dir)
    mk_dir(testset_dir)

    bboxes_mat = scipy.io.loadmat(aflw_pifa_mat[0])
    kps_mat = scipy.io.loadmat(aflw_pifa_mat[1])
    bboxesT = bboxes_mat['bboxesT']
    bboxesTr = bboxes_mat['bboxesTr']
    phisT = kps_mat['phisT']
    phisTr = kps_mat['phisTr']
    im_dirT = aflw_pifa_image_dir[0]
    im_dirTr = aflw_pifa_image_dir[1]
    for im_dir, bboxes, phis, save_dir, im_sum in zip([im_dirT, im_dirTr],
                                                       [bboxesT, bboxesTr],
                                                       [phisT, phisTr],
                                                       [testset_dir, trainset_dir],
                                                       [1299, 3901]):
        reported_im_names = []
        for im_name, im_kps, im_mask, im_box in zip(range(1, im_sum+1), phis[:, :68], phis[:, 68:], bboxes):
            # save image
            im_path = os.path.join(save_dir, str(im_name) + '.jpg')
            ori_im_path = os.path.join(im_dir, str(im_name) + '.jpg')
            pts_path = im_path.replace('jpg', 'pts')
            pickle_path = im_path.replace('jpg', 'pkl')
            if not os.path.exists(ori_im_path):
                print 'invalid path'
                exit(1)
            shutil.copy(ori_im_path, im_path)

            # visual_kp_debug(im_path, im_kps, [im_box[0], im_box[1],
            #                                   im_box[0] + im_box[2],
            #                                   im_box[1] + im_box[3]])

            # save pts
            with open(pts_path, 'w') as f:
                f.write('version: 1\n')
                f.write('n_points:  34\n')
                f.write('{\n')
                im_kps = im_kps.reshape([2, -1]).transpose()
                for im_kp in im_kps:
                    f.write('%f %f' % (im_kp[0], im_kp[1]) + '\n')
                f.write('}')

            reported_im_names.append(im_name)
            print 'reported num: %d/%d' % (len(reported_im_names), im_sum)

            # save other information
            with open(pickle_path, 'wb') as f:
                im_box = [im_box[0], im_box[1],
                          im_box[0] + im_box[2],
                          im_box[1] + im_box[3]]
                info = {'box': np.array(im_box)}
                info['mask'] = np.array(im_mask)
                pickle.dump(info, f)
            # debug
            # with open(pickle_path, 'rb') as f:
            #     newinfo = pickle.load(f)
            #     pass


def transform_COFW_to_pts(db_name='cofw'):
    trainset_dir = os.path.join(pts_savedir, db_name, 'trainset')
    testset_dir = os.path.join(pts_savedir, db_name, 'testset')
    mk_dir(trainset_dir)
    mk_dir(testset_dir)

    test_mat = h5py.File(cofw_mat[0])
    train_mat = h5py.File(cofw_mat[1])
    bboxesT = np.transpose(test_mat['bboxesT'])
    bboxesTr = np.transpose(train_mat['bboxesTr'])
    phisT = np.transpose(test_mat['phisT'])
    phisTr = np.transpose(train_mat['phisTr'])
    imsT = np.transpose(test_mat['IsT'])
    imsTr = np.transpose(train_mat['IsTr'])
    for ims, bboxes, phis, save_dir, im_sum, mat in zip([imsT, imsTr],
                                                   [bboxesT, bboxesTr],
                                                   [phisT, phisTr],
                                                   [testset_dir, trainset_dir],
                                                   [507, 1345], [test_mat, train_mat]):
        reported_im_names = []
        for im_name, im, im_kps, im_mask, im_box in zip(range(1, im_sum+1), ims, phis[:, :58], phis[:, 58:], bboxes):
            # save image
            im_path = os.path.join(save_dir, str(im_name) + '.jpg')
            pts_path = im_path.replace('jpg', 'pts')
            pickle_path = im_path.replace('jpg', 'pkl')
            im = np.transpose(mat[im[0]][:])
            scipy.misc.imsave(im_path, im)
            # visual_kp_debug(im_path, im_kps, [im_box[0], im_box[1],
            #                                   im_box[0] + im_box[2],
            #                                   im_box[1] + im_box[3]])
            # save pts
            with open(pts_path, 'w') as f:
                f.write('version: 1\n')
                f.write('n_points:  29\n')
                f.write('{\n')
                im_kps = im_kps.reshape([2, -1]).transpose()
                for im_kp in im_kps:
                    f.write('%f %f' % (im_kp[0], im_kp[1]) + '\n')
                f.write('}')

            reported_im_names.append(im_name)
            print 'reported num: %d/%d' % (len(reported_im_names), im_sum)

            # save other information
            with open(pickle_path, 'wb') as f:
                im_box = [im_box[0], im_box[1],
                          im_box[0] + im_box[2],
                          im_box[1] + im_box[3]]
                info = {'box': np.array(im_box)}
                pickle.dump(info, f)
            # debug
            # with open(pickle_path, 'rb') as f:
            #     newinfo = pickle.load(f)
            #     pass


def excute_datasets(idx, datatype):
    #root_path = os.getcwd()
    f = open('../DB/object/voc_wider/ImageSets/Main/'+datatype+'.txt', 'w')
    mat = h5py.File('../DB/face/wider/wider_face_'+datatype+'.mat', 'r')
    file_list = mat['file_list'][:]
    event_list = mat['event_list'][:]
    bbx_list = mat['face_bbx_list'][:]
    # sum = 0
    for i in range(file_list.size):
        file_list_sub = mat[file_list[0,i]][:]
        bbx_list_sub = mat[bbx_list[0, i]][:]
        event_value = ''.join(chr(x) for x in mat[event_list[0,i]][:])
        for j in range(file_list_sub.size):
            root = '../DB/face/wider/'+datatype+'/'+event_value+'/'
            filename = root + ''.join([chr(x) for x in mat[file_list_sub[0, j]][:]])+'.jpg'
            im = io.imread(filename)
            head = headstr % (idx, im.shape[1], im.shape[0], im.shape[2])
            bboxes = mat[bbx_list_sub[0, j]][:]
            # for k in range(bboxes.shape[1]):
            #     if bboxes[2, k] == 0 or bboxes[3, k] == 0:
            #         print "%f %f %f %f" % (bboxes[0,k],bboxes[1,k], bboxes[0,k]+bboxes[2,k],bboxes[1,k]+bboxes[3,k])
            #         sum += 1
            bboxes = bbox_filter(bboxes, filename, im.shape[1], im.shape[0], idx)
            if bboxes.shape[1] == 0:
                print '[+] Discard:', filename
                continue
            objs = ''.join([objstr % ('face', \
                   bboxes[0,k],bboxes[1,k], bboxes[0,k]+bboxes[2,k],bboxes[1,k]+bboxes[3,k]) \
                   for k in range(bboxes.shape[1])])
            writexml(idx, head, objs, tailstr)
            shutil.copyfile(filename, '../DB/object/voc_wider/JPEGImages/%06d.jpg' % (idx))
            f.write('%06d\n' % (idx))
            idx += 1
            # if idx == 1161:
            #     print 1
            print idx
    # print 'when h or w is 0, sum: %d' % sum
    f.close()
    return idx


def save_label_list(dbname, type):
    label_dir = os.path.join(savedir, dbname, type, 'labels')
    save_path = os.path.join(label_savedir, '%s_%s_lbl_list.txt' % (dbname, type))
    with open(save_path, 'w') as f:
        for name in os.listdir(label_dir):
            f.write(label_dir + '/%s\n' % name)


def show_facial_kps(dbname, type):
    img_dir = os.path.join(savedir, dbname, type, 'images')
    for name in os.listdir(img_dir):
        img_path = img_dir + '/%s' % name
        fi = FaceImage(img_path)
        fi.vis()


def visual_kp_debug(img, kps, boxs=None):
    import cv2
    import matplotlib.pyplot as plt

    kps = kps.reshape([2, -1]).transpose()
    im = cv2.imread(img)
    f = plt.figure(figsize=(10, 6))
    subplot = f.add_subplot(111)
    plt.imshow(im[:, :, ::-1])
    if boxs is not None:
        draw_bbox(subplot, boxs)
    plt.plot(kps[:, 0], kps[:, 1], 'go', ms=1.5, alpha=1)
    plt.close('all')

if __name__ == '__main__':
    # path_to_images = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/300w-train'
    # path_to_images = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/aflw-full/testset-frontal'
    # path_to_images = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/aflw-pifa/trainset'
    # path_to_images = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/cofw/testset'
    # transform_pts(path_to_images, '300w(re)', 'trainval')  # testset-frontal
    # transform_pts(path_to_images, 'aflw-full', 'testset-frontal', box_c=1)
    # transform_pts(path_to_images, 'aflw-pifa', 'trainset', box_c=1)
    # transform_pts(path_to_images, 'cofw', 'testset', box_c=1)

    # save_label_list('300w', 'trainval')
    # save_label_list('300w(re)', 'trainval')
    # save_label_list('300w-c1', 'trainval')
    # save_label_list('300w-c1(ti)', 'trainval')
    # save_label_list('300w-c(re)', 'trainval')
    # save_label_list('aflw-full', 'testset-frontal')
    # save_label_list('aflw-pifa', 'trainset')
    # save_label_list('cofw', 'testset')

    # show_facial_kps('300w-c', 'trainval')
    # show_facial_kps('afw', 'trainval')

    # generate AFLW pts
    # transform_AFLW_full_to_pts()
    # transform_AFLW_pifa_to_pts()
    # transform_COFW_to_pts()
    print 'done'