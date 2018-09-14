#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import _init_ms_paths
import menpofit
import menpo.io as mio
from menpo.feature import fast_dsift, igo
from menpo.visualize import print_progress
from menpofit.aam import HolisticAAM
from menpowidgets import visualize_images
# from menpowidgets import visualize_pointclouds
from pathlib import Path
from menpofit.fitter import noisy_shape_from_shape
from menpo.transform import AlignmentAffine
from menpofit.modelinstance import OrthoPDM
import matplotlib.pyplot as plt
from menpo.transform import compositions, AlignmentSimilarity
import os
from mylab.tool import mk_dir

support_types = ['afw', 'lfpw', 'helen']

def test():
    img = mio.import_image('/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/aflw-full/testset/0_image00002_1.jpg')

    if img.n_channels != 1:
        img = img.as_greyscale()

    img.landmarks['face'] = mio.import_landmark_file('/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/temp/indoor_001.pts')

    # objects return copies rather than mutating self, so we can chain calls
    img = (img.crop_to_landmarks(group='face', boundary=10)
           .rescale_landmarks_to_diagonal_range(100, group='face'))

    # now lets take an image feature...
    img = fast_dsift(img)

    # ...and extract the vector of pixels contained in the
    # convex hull of the face...
    vector = img.as_masked().constrain_mask_to_landmarks(group='face').as_vector()

    print(type(vector), vector.shape)
    # output: <class 'numpy.ndarray'> (3801,)

def fit_pre(image_paths):
    import menpo.io as mio
    from menpowidgets import visualize_images
    from menpo.visualize import print_progress

    training_images = []
    for image_path in image_paths:
        for img in print_progress(mio.import_images(image_path, verbose=True)):
            # convert to greyscale
            if img.n_channels == 3:
                img = img.as_greyscale()
            # crop to landmarks bounding box with an extra 20% padding
            img = img.crop_to_landmarks_proportion(0.2)
            # rescale image if its diagonal is bigger than 400 pixels
            d = img.diagonal()
            if d > 400:
                img = img.rescale(400.0 / d)
#             %matplotlib inline
#             visualize_images(img)
            # append to list
            training_images.append(img)
    return training_images

def train(training_images):
    aam = HolisticAAM(training_images, reference_shape=None,
                      diagonal=150, scales=(0.5, 1.0),
                      holistic_features=igo, verbose=True)
    print(aam)
    return aam

def train_AAM(training_images, feature=igo):
    aam = HolisticAAM(training_images, reference_shape=None,
                      diagonal=150, scales=(0.5, 1.0),
                      holistic_features=feature, verbose=True,
                      max_shape_components=20, max_appearance_components=150)
    print(aam)
    return aam

def fitter_AAM(aam):
    from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional

    fitter = LucasKanadeAAMFitter(aam, lk_algorithm_cls=WibergInverseCompositional,
                                  n_shape=[5, 20], n_appearance=[30, 150])
    return fitter

def test_AAM(fitter, images):
    for image in images:
        image = mio.import_image(image)
        image = image.as_greyscale()
        initial_bbox = image.landmarks['PTS'].bounding_box()
        gt_shape = image.landmarks['PTS'].lms
        initial_shape = noisy_shape_from_bounding_box(gt_shape, gt_shape.bounding_box())
        image.landmarks['boundingbox'] = initial_bbox
        image.landmarks['init_shape'] = initial_shape
        image.view_landmarks(group='boundingbox', line_colour='red',
                             render_markers=False, line_width=4)
        image.view_landmarks(group='init_shape')
        # fit image
        result = fitter.fit_from_bb(image, initial_bbox, max_iters=[15, 5],
                                    gt_shape=image.landmarks['PTS'].lms)
        # print result
        print(result)

        # fit image
        result1 = fitter.fit_from_shape(image, initial_shape, max_iters=[15, 5],
                                    gt_shape=image.landmarks['PTS'].lms)
        # print result
        print(result1)

        result.view(render_initial_shape=True)

def pca(path_to_images, max_n_components=None):
    path_to_lfpw = Path(path_to_images)

    training_shapes = []
    for lg in print_progress(mio.import_landmark_files(path_to_lfpw / '*.pts', verbose=True)):
        training_shapes.append(lg)  # lg['all']

    shape_model = OrthoPDM(training_shapes, max_n_components=max_n_components)
    print(shape_model)
    # visualize_pointclouds(training_shapes)
    # instance = shape_model.similarity_model.instance([100., -300., 0., 0.])
    # instance.view(render_axes=False)
    return shape_model

def pca_image(images, max_n_components=None):

    training_shapes = []
    for image in images:
        training_shapes.append(image.landmarks['PTS'])  # lg['all']

    shape_model = OrthoPDM(training_shapes, max_n_components=max_n_components)
    print(shape_model)
    # visualize_pointclouds(training_shapes)
    # instance = shape_model.similarity_model.instance([100., -300., 0., 0.])
    # instance.view(render_axes=False)
    return shape_model

def reconstructByPca(path_to_images):
    shape_model = pca(path_to_images)

    # Import shape
    shape = mio.import_builtin_asset.einstein_pts().lms

    # Find the affine transform that normalizes the shape
    # with respect to the mean shape
    transform = AlignmentAffine(shape, shape_model.model.mean())

    # Normalize shape and project it
    normalized_shape = transform.apply(shape)
    weights = shape_model.model.project(normalized_shape)
    print("Weights: {}".format(weights))

    # Reconstruct the normalized shape
    reconstructed_normalized_shape = shape_model.model.instance(weights)

    # Apply the pseudoinverse of the affine tansform
    reconstructed_shape = transform.pseudoinverse().apply(reconstructed_normalized_shape)

    # Visualize
    plt.subplot(121)
    shape.view(render_axes=False, axes_x_limits=0.05, axes_y_limits=0.05)
    plt.gca().set_title('Original shape')
    plt.subplot(122)
    reconstructed_shape.view(render_axes=False, axes_x_limits=0.05, axes_y_limits=0.05)
    plt.gca().set_title('Reconstructed shape')

def affine():
    import menpo.io as mio
    takeo = mio.import_builtin_asset.takeo_ppm()
    # Use a bounding box of ground truth landmarks to create template
    takeo.landmarks['bounding_box'] = takeo.landmarks['PTS'].lms.bounding_box()
    template = takeo.crop_to_landmarks(group='bounding_box', boundary=10)
    template.view()
    from menpofit.lk import LucasKanadeFitter, InverseCompositional, SSD
    from menpo.feature import no_op

    fitter = LucasKanadeFitter(template, group='bounding_box',
                               algorithm_cls=InverseCompositional, residual_cls=SSD,
                               scales=(0.5, 1.0), holistic_features=no_op)
    print fitter

    from menpofit.fitter import noisy_shape_from_bounding_box

    gt_bb = takeo.landmarks['bounding_box'].lms
    # generate perturbed bounding box
    init_bb = noisy_shape_from_bounding_box(fitter.reference_shape, gt_bb,
                                            noise_percentage=0.1,
                                            allow_alignment_rotation=True)
    # fit image
    result = fitter.fit_from_bb(takeo, init_bb, gt_shape=gt_bb, max_iters=80)

    # result.view()

    from menpowidgets import visualize_images
    visualize_images(fitter.warped_images(result.image, result.shapes))

def areaOflandmarks(landmarks):
    corner_a, corner_b = landmarks.bounds()
    return (corner_b - corner_a).prod()

def affine_test(path_to_images):
    # create pca model based on training set
    shape_model = pca(path_to_images)
    # create test image
    # testImage = mio.import_builtin_asset.takeo_ppm()
    testImage = mio.import_image('/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/aflw-full/testset/0_image00002_1.jpg')

    # Find the affine transform that normalizes the shape
    # with respect to the mean shape
    # shape = testImage.landmarks['PTS']
    # transform = AlignmentAffine(shape, shape_model.model.mean())

    # image change size to adapt the scale of kp to that of target_kp
    tmp_image = testImage.rescale_to_pointcloud(shape_model.model.mean(),
                                                group='PTS')

    transform = AlignmentAffine(testImage.landmarks['PTS'], tmp_image.landmarks['PTS'])
    new_shape = transform.apply(testImage.landmarks['PTS'])  # equal to kp of tmp

    # warp image
    # new_image = testImage.warp_to_shape(tmp_image.shape, transform.pseudoinverse(),
    #               warp_landmarks=True, order=1,
    #               mode='nearest',
    #               return_transform=False)
    #
    # plt.subplot(131)
    # testImage.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
    #                          marker_size=4, render_axes=True)
    # plt.gca().set_title('Original image')
    # plt.subplot(132)
    # new_image.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
    #                          marker_size=4, render_axes=True)
    # plt.gca().set_title('Rescale image')
    # plt.subplot(133)
    # tmp_image.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
    #                          marker_size=4, render_axes=True)
    # plt.gca().set_title('Template image')

    # create a noise shape
    # noisy_shape = noisy_shape_from_shape(testImage.landmarks['PTS'], testImage.landmarks['PTS'],
    #                                         noise_percentage=0.2,
    #                                         allow_alignment_rotation=True)
    # transform = AlignmentAffine(testImage.landmarks['PTS'], noisy_shape)

    # similarity = AlignmentSimilarity(testImage.landmarks['PTS'],
    #                                               testImage.landmarks['PTS'],
    #                                               rotation=True)
    s = compositions.scale_about_centre(testImage.landmarks['PTS'], 1)
    r = compositions.rotate_ccw_about_centre(testImage, 90)
    t = compositions.Translation([0, 0], testImage.n_dims)
    # transform = similarity.compose_after(t.compose_after(s.compose_after(r)))
    transform = t.compose_after(s.compose_after(r))
    # new_shape = transform.apply(testImage.landmarks['PTS'])

    # warp image
    new_image = testImage.warp_to_shape(testImage.shape, transform.pseudoinverse(),
                                        warp_landmarks=True, order=1,
                                        mode='nearest',
                                        return_transform=False)
    plt.subplot(121)
    testImage.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
                             marker_size=4, render_axes=True)
    plt.gca().set_title('Original image')
    plt.subplot(122)
    new_image.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
                             marker_size=4, render_axes=True)
    plt.gca().set_title('Rescale image')
    plt.close('all')

def affine_enhance(path_to_images, save_dir=None, scales=[1], rotations=[0], translations=[[0, 0]], mean_shape=1):
    if save_dir is not None:
        mk_dir(save_dir, 0)
    # load training images
    train_images = []
    for path_to_image in path_to_images:
        for img in print_progress(mio.import_images(path_to_image, verbose=True)):
            train_images.append(img)
    print 'sum of training data: %d' % len(train_images)
    # create pca model based on training set
    # shape_model = pca(path_train_images)
    shape_model = pca_image(train_images)
    excepted_num = len(scales)*len(rotations)*len(translations)*len(train_images)
    completed_num = 0
    for train_img in train_images:
        if mean_shape:
            transform = AlignmentAffine(train_img.landmarks['PTS'], shape_model.model.mean())
            [r1, s, r2, t] = transform.decompose()
            # transform = r2.compose_after(s.compose_after(r1))
            transform = r2.compose_after(r1)
            rotation_shape = transform.apply(train_img.landmarks['PTS'])
            offset = train_img.landmarks['PTS'].centre() - rotation_shape.centre()
            t = compositions.Translation(offset, train_img.n_dims)
            transform = t.compose_after(r2.compose_after(r1))
            normal_image = train_img.warp_to_shape(train_img.shape, transform.pseudoinverse(),
                                                warp_landmarks=True, order=1,
                                                mode='nearest',
                                                return_transform=False)
        else:
            normal_image = train_img
        for scale in scales:
            for rotation in rotations:
                for translation in translations:
                    s = compositions.scale_about_centre(normal_image.landmarks['PTS'], scale)
                    r = compositions.rotate_ccw_about_centre(normal_image, rotation)
                    t = compositions.Translation(translation, normal_image.n_dims)
                    transform = t.compose_after(s.compose_after(r))

                    # warp image
                    new_image = normal_image.warp_to_shape(normal_image.shape, transform.pseudoinverse(),
                                                        warp_landmarks=True, order=1,
                                                        mode='nearest',
                                                        return_transform=False)
                    # plt.subplot(121)
                    # normal_image.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
                    #                          marker_size=4, render_axes=True)
                    # plt.gca().set_title('Original image')
                    # plt.subplot(122)
                    # new_image.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
                    #                          marker_size=4, render_axes=True)
                    # plt.gca().set_title('Rescale image')
                    # plt.close('all')

                    # save enhanced image with lable
                    img_suffix = new_image.path.suffix
                    lb_suffix = '.pts'
                    dataType = filter(lambda x: x in str(new_image.path), support_types)[0]
                    new_image_name = '%s_' % dataType + new_image.path.name.split('.')[0] + '_s%s_r%s_x%s_y%s' % (str(scale), str(rotation), str(translation[0]), str(translation[1]))
                    img_path = os.path.join(save_dir, new_image_name+img_suffix)
                    lb_path = os.path.join(save_dir, new_image_name+lb_suffix)
                    mio.export_image(new_image, img_path, overwrite=True)
                    mio.export_landmark_file(new_image.landmarks['PTS'], lb_path, overwrite=True)

                    # plt.subplot(121)
                    # new_image.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
                    #                             marker_size=4, render_axes=True)
                    # plt.gca().set_title('new image')
                    # save_image = mio.import_image(img_path)
                    # plt.subplot(122)
                    # save_image.view_landmarks(marker_face_colour='white', marker_edge_colour='black',
                    #                          marker_size=4, render_axes=True)
                    # plt.gca().set_title('saved image')
                    # plt.close('all')
                    completed_num = completed_num + 1
                    print 'completed: %d/%d' % (completed_num, excepted_num)

def img_pre(path_to_images, save_dir=None, propotion=0.2, scale=400.0, greyscale=False):
    import menpo.io as mio
    from menpo.visualize import print_progress

    if save_dir is not None:
        mk_dir(save_dir, 0)

    for image_path in path_to_images:
        for img in print_progress(mio.import_images(image_path, verbose=True)):
            if greyscale:
                # convert to greyscale
                if img.n_channels == 3:
                    img = img.as_greyscale()
            # crop to landmarks bounding box with an extra 20% padding
            re_img = img.crop_to_landmarks_proportion(propotion)

            # # rescale image if its diagonal is bigger than 400 pixels
            # d = img.diagonal()
            # if d > scale:
            #     img = img.rescale(scale / d)

            # save enhanced image with lable
            img_suffix = img.path.suffix
            lb_suffix = '.pts'
            new_image_name = '%s' % img.path.name.split('.')[0]
            img_path = os.path.join(save_dir, new_image_name + img_suffix)
            lb_path = os.path.join(save_dir, new_image_name + lb_suffix)
            mio.export_image(re_img, img_path, overwrite=True)
            mio.export_landmark_file(re_img.landmarks['PTS'], lb_path, overwrite=True)

            # debug
            # plt.subplot(121)
            # plt.gca().set_title('Original image')
            # img.view_landmarks(group='PTS', render_axes=True)
            # plt.subplot(122)
            # plt.gca().set_title('New image')
            # re_img.view_landmarks(group='PTS', render_axes=True)
            # plt.close('all')

    # return training_images

def deformabel(path_to_images):
    from pathlib import Path
    import menpo.io as mio

    path_to_lfpw = Path(path_to_images)

    image = mio.import_image(path_to_lfpw / 'image_0004.png')
    image = image.crop_to_landmarks_proportion(0.5)

    template = mio.import_image(path_to_lfpw / 'image_0018.png')
    template = template.crop_to_landmarks_proportion(0.5)

    template.view_landmarks(1, marker_face_colour='white', marker_edge_colour='black',
                            marker_size=4)

    from menpo.visualize import print_progress

    training_shapes = []
    for lg in print_progress(mio.import_landmark_files(path_to_lfpw / '*.pts', verbose=True)):
        training_shapes.append(lg)

    from menpofit.atm import HolisticATM
    from menpo.feature import igo

    atm = HolisticATM(template, training_shapes, group='PTS',
                      diagonal=180, scales=(0.25, 1.0),
                      holistic_features=igo, verbose=True)
    from menpofit.atm import LucasKanadeATMFitter, InverseCompositional

    fitter = LucasKanadeATMFitter(atm,
                                  lk_algorithm_cls=InverseCompositional, n_shape=[5, 15])

    # from menpodetect import load_dlib_frontal_face_detector
    #
    # # Load detector
    # detect = load_dlib_frontal_face_detector()
    #
    # # Detect
    # bboxes = detect(image)
    # print("{} detected faces.".format(len(bboxes)))

    initial_bbox = image.landmarks['PTS'].bounding_box()

    # # View
    # if len(bboxes) > 0:
    #     image.view_landmarks(group='dlib_0', line_colour='white',
    #                          render_markers=False, line_width=3)

    # initial bbox
    # initial_bbox = bboxes[0]

    # fit image
    result = fitter.fit_from_bb(image, initial_bbox, max_iters=20,
                                gt_shape=image.landmarks['PTS'].lms)

    # print result
    print(result)

    result.view(2, render_initial_shape=True)

if __name__ == '__main__':
    # path_to_images = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/lfpw/trainset']
    # path_to_images = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/afw']
    # path_to_images = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/helen/trainset']
    # test_images = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/300w_cropped/01_Indoor/indoor_110.png']
    # training_images = fit_pre(path_to_images)
    # aam = train_AAM(training_images)
    # fitter = fitter_AAM(aam)
    # test_AAM(fitter, test_images)

    # save_dir = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/300w-c2'  # 300w-c
    # affine_enhance(path_to_images, save_dir=save_dir, rotations=[0], mean_shape=0)  # [0, 15, -15] 30
    # path_to_images = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/aflw-full/testset-frontal']
    # affine_test(path_to_images[0])
    # reconstructBypca(path_to_images[0])
    # test()

    # save_dir = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/300w-v2(p0.2)/01_Indoor'
    # path_to_images = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/300w_cropped/01_Indoor']
    save_dir = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/otherDB/300w-v2(p0.2)/02_Outdoor'
    path_to_images = ['/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/300w_cropped/02_Outdoor']

    img_pre(path_to_images, save_dir=save_dir, propotion=0.2)
    print 'done'