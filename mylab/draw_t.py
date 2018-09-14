import numpy as np
from pylab import *
import matplotlib.patches as patches
import numpy.random as npr
from PIL import Image
import os, sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def __get_ellipse_points(major, minor, angle, x, y):
    '''

    :param major: ellipse major value (half)
    :param minor: ellipse major value (half)
    :param angle: not in degree. If in degree, convert it by: -angle*np.pi/180.0
    :param x: center x
    :param y: center y
    :return: set of points with shape<721,2>
    '''
    pts = np.zeros((721, 2))  # 721 xy-points
    # beta = -angle*np.pi/180.0
    beta = angle
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.radians(np.r_[0.0:360.0:1j * (721)])
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    pts[:, 0] = x + (major * cos_alpha * cos_beta - minor * sin_alpha * sin_beta)
    pts[:, 1] = y + (major * cos_alpha * sin_beta + minor * sin_alpha * cos_beta)

    return pts


def ellipse2box(major, minor, angle, x, y):
    '''

    :param major: ellipse major value (half)
    :param minor: ellipse major value (half)
    :param angle: not in degree. If in degree, convert it by: -angle*np.pi/180.0
    :param x: center x
    :param y: center y
    :return: list type: [xmin, ymin, xmax, ymax]
    '''
    pts = __get_ellipse_points(major, minor, angle, x, y)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def draw_ellipse(subplot, elipses, color='b', linewidth=3):
    '''

    :param subplot: a subplot
    :param elipses:
        ellipses in "major, minor, angle, x, y" order
        expecting a list with length 5; OR a np.array with shape <n,5>
    :param color: line color
    :param linewidth: line width
    :return:
    '''
    elipses = np.array(elipses)
    if len(elipses.shape) == 1: elipses = np.array([elipses])
    for e in elipses:
        pts = __get_ellipse_points(*e)
        subplot.plot(pts[:, 0], pts[:, 1], '{}'.format(color))


def draw_bbox(subplot, bboxes, color='cyan', linewidth=2, alpha=1, fontsize=12, extra_texts=None, fontcolor=None):
    # bbox: xmin, ymin, xmax, ymax
    #
    '''

    :param subplot:  a subplot
    :param bboxes:
        bounding boxes in "xmin, ymin, xmax, ymax" order
        expecting a list with length 4; OR a np.array with shape <n,4>
    :param color: line color
    :param linewidth: line width
    :return:
    '''

    bboxes = np.array(bboxes)

    if len(bboxes.shape) == 1: bboxes = np.array([bboxes])
    idx = -1
    for b in bboxes:
        idx +=1
        x = b[0];
        y = b[1]
        w = b[2] - x
        h = b[3] - y
        subplot.add_patch(patches.Rectangle(
            (x, y),  # (x,y)
            w,  # width
            h,  # height
            fill=0,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha
        ))
        if extra_texts is not None:
            assert len(extra_texts) == bboxes.shape[0]
            score = extra_texts[idx]
            if score == '' or score==None: continue
            # fontsize = w/(len(extra_texts[0])+0.0)
            # if fontcolor is None: fontcolor = color
            # subplot.text((b[0] + b[2]) / 2, b[3] + fontsize / 2, score,
            #              horizontalalignment='center',
            #              verticalalignment='center',
            #              color=fontcolor,
            #              fontsize=fontsize,
            #              alpha=alpha,
            #              bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

            if fontcolor is None: fontcolor = color
            subplot.text((b[0] + b[2]) / 2, b[3] + fontsize / 2, score,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=fontcolor,
                         fontsize=fontsize,
                         alpha=alpha,
                         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})


def fcn_color_map(im, fg, bg, out, LA=1, RA=0):  # left alpha/right alpha
    '''

    :param im: an opened original image, assuming it has shape <A,B>
    :param fg: foreground heatmap with shape <A,B>
    :param bg: background heatmap with shape <A,B>
    :param out: argmaxed mask with shape <A,B>
    :param LA: alpha value of the image [0,1]==>[nothing, no-alpha]
    :param RA: alpha value of masks [0,1]==>[nothing, no-alpha]
    :return:
    '''
    out = out[..., np.newaxis]
    out = np.tile(out, (1, 1, 3))
    w = fg.shape[1];
    h = fg.shape[0]
    min_max = lambda x: ((x - x.min()) / (x.max() - x.min()))
    fg = min_max(fg)
    bg = min_max(bg)
    if w > h:
        f = figure(figsize=(16, 27))
    else:
        f = figure(figsize=(9, 32))
    # ----------------------#
    f.add_subplot(521);
    imshow(fg);
    title('FG')
    f.add_subplot(522);
    imshow(im, alpha=LA);
    imshow(fg, alpha=RA)
    # ----------------------#
    f.add_subplot(523);
    imshow(bg);
    title('BG')
    f.add_subplot(524);
    imshow(im, alpha=LA);
    imshow(bg, alpha=RA)
    # ----------------------#
    f.add_subplot(525);
    imshow(fg - bg);
    title('FG-BG')
    f.add_subplot(526);
    imshow(im, alpha=LA);
    imshow(fg - bg, alpha=RA)
    # ----------------------#
    f.add_subplot(527);
    imshow(out);
    title('ArgMax')
    f.add_subplot(528);
    imshow(im, alpha=LA);
    imshow(out, alpha=RA)
    # ----------------------#
    f.add_subplot(529);
    imshow(fg + bg);
    title('FG+BG')
    f.add_subplot(5, 2, 10);
    imshow(im, alpha=LA);
    imshow(fg + bg, alpha=RA)


def fcn_3d(fg, figsize=(9, 9)):
    '''

    :param fg: foreground heatmap with shape <A,B>
    :param figsize: figure size, a tuple, default (9,9)
    :return:
    '''
    fig = figure(figsize=figsize)
    X, Y = np.meshgrid(np.linspace(0, fg.shape[1], fg.shape[1]), np.linspace(0, fg.shape[0], fg.shape[0]))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, fg, cmap=cm.hot, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
