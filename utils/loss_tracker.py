#!/usr/bin/env python
import matplotlib.pylab as plt
import numpy as np
import re
import math

def get_log_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l for l in lines if 'Im' != l[0:2] ]
    return ''.join(lines)

def get_iter_loss(data):
    loss = re.findall(' loss = [0-9]*.[0-9]*\n', data)
    iter = re.findall('] Iteration [0-9]*', data)
    loss = [float(i.strip(' loss = ')) for i in loss]
    iter = [int(i.strip('] Iteration ')) for i in iter][::2]
    return iter,loss

def median(midlist):
    midlist.sort()
    lens = len(midlist)
    if lens % 2 != 0:
        midl = (lens / 2)
        res = midlist[midl]
    else:
        odd = (lens / 2) -1
        ev = (lens / 2)
        res = float(midlist[odd] + midlist[ev]) / float(2)
    return res

def cal_mid_line(iter, loss):
    mid_iter = []
    mid_loss = []

    iter_window_size = 1000/20
    step_size= 800/20
    lower_idx = 0

    lens = len(iter)
    while True:
        if(lower_idx >= lens):
            upper_idx = lens
        upper_idx = lower_idx + iter_window_size
        sub_loss = loss[lower_idx: upper_idx]
        m = median(sub_loss)
        mid_iter.append(iter[lower_idx]+iter_window_size*10)
        mid_loss.append(m)

        lower_idx = lower_idx+step_size
        if(upper_idx >= lens):
            break
    # sub_loss = loss[-window_size:]
    # m = median(sub_loss)
    # mid_iter.append(iter[-1])
    # mid_loss.append(m)
    return mid_iter, mid_loss

def plot_loss(iter, loss, update_times=0, update_interval=20):
    plt.ion()
    plt.clf()
    mid_iter, mid_loss = cal_mid_line(iter, loss)

    plt.plot(mid_iter, mid_loss, 'r-', linewidth=3)
    plt.plot(iter, np.ones(len(iter)),'g.',linewidth=1)
    # plt.plot(iter, np.ones(len(iter))*0.5, 'g.', linewidth=1)

    plt.plot(iter, loss, 'b-', alpha=0.4)
    plt.title("Iteration:{}   UpdateTimes:{} ({}s)".format(max(iter), update_times, update_interval))

    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.yticks(np.arange(0, 10, 0.5))
    plt.show()
    plt.savefig('/tmp/fig.png')
    for i in range(update_interval):
        plt.grid()
        plt.title("Iteration:{}   UpdateTimes:{} ({}s)".format(max(iter), update_times, update_interval-i))
        plt.pause(1)
        plt.grid()

def get_loss_data(loss_path):
    with open(loss_path, 'r') as f:
        lines = f.readlines()
    return ''.join(lines)

def get_iter_loss_face(data, lossTypeStr='loss'):
    loss = re.findall(' %s [0-9]*.[0-9]*' % lossTypeStr, data)
    iter = re.findall('iter_[0-9]*', data)
    loss = [float(i.strip(' %s ' % lossTypeStr)) for i in loss][::20]
    iter = [int(i.strip('iter_')) for i in iter][::20]
    return iter, loss

def plot_loss_face(iter, loss, lossTypeStr, savePath=''):
    plt.ion()
    # plt.clf()
    mid_iter, mid_loss = cal_mid_line(iter, loss)

    plt.plot(mid_iter, mid_loss, 'r-', linewidth=3)
    plt.plot(iter, np.ones(len(iter)),'g.',linewidth=1)
    # plt.plot(iter, np.ones(len(iter))*0.5, 'g.', linewidth=1)

    plt.plot(iter, loss, 'b-', alpha=0.4)
    plt.title("Iteration:{}  Loss:{}".format(max(iter), lossTypeStr))

    plt.xlabel("iterations")
    plt.ylabel("loss")
    ymax = np.mean(loss)*5
    plt.yticks(np.linspace(0, ymax, 20))
    plt.ylim(0, ymax)
    plt.show()
    plt.grid()
    if savePath != '':
        plt.savefig(savePath)
    plt.pause(1)
    # for i in range(update_interval):
    #     plt.grid()
    #     plt.title("Iteration:{}   UpdateTimes:{} ({}s)".format(max(iter), update_times, update_interval-i))
    #     plt.pause(1)
    #     plt.grid(1)

def plot_loss_face_1(iter, loss, lossTypeStr, savePath=''):
    plt.ion()
    # plt.clf()
    mid_iter, mid_loss = cal_mid_line(iter, loss)

    plt.plot(mid_iter, mid_loss, 'r-', linewidth=3)
    plt.plot(iter, np.ones(len(iter)),'g.',linewidth=1)
    # plt.plot(iter, np.ones(len(iter))*0.5, 'g.', linewidth=1)

    plt.plot(iter, loss, 'b-', alpha=0.4)
    plt.title("Iteration:{}  Loss:{}".format(max(iter), lossTypeStr))

    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.yticks(np.linspace(0, sum(loss)/len(loss)*3, 20))
    plt.ylim(0, sum(loss)/len(loss)*3)
    plt.show()
    plt.grid()
    if savePath != '':
        plt.savefig(savePath)
    plt.pause(1)
    # for i in range(update_interval):
    #     plt.grid()
    #     plt.title("Iteration:{}   UpdateTimes:{} ({}s)".format(max(iter), update_times, update_interval-i))
    #     plt.pause(1)
    #     plt.grid(1)

def run_plot_loss(path, lossStrList):
    paths = path.split('/')
    savePath = '/'.join(paths[:-1]) + '/VRLoss_%s.png' % (paths[-1][5:][:-4])
    for index, lossStr in enumerate(lossStrList):
        # paths = path.split('/')
        # savePath = '/'.join(paths[:-1]) + '/V%s_%s.png' % (lossStr, paths[-1][5:][:-4])
        data = get_loss_data(path)
        iter, loss = get_iter_loss_face(data, lossStr)
        plt.subplot(2, int(math.ceil(len(lossStrList)/2.0)), index+1)
        plot_loss_face(iter, loss, lossStr)
        plt.tight_layout()
    plt.savefig(savePath)
    plt.close('all')

def run_plot_loss_1(path, lossStrList):
    paths = path.split('/')
    savePath = '/'.join(paths[:-1]) + '/VRLoss_%s.png' % (paths[-1][5:][:-4])
    plt.figure(figsize=(20, 20))
    for index, lossStr in enumerate(lossStrList):
        # paths = path.split('/')
        # savePath = '/'.join(paths[:-1]) + '/V%s_%s.png' % (lossStr, paths[-1][5:][:-4])
        data = get_loss_data(path)
        iter, loss = get_iter_loss_face(data, lossStr)
        plt.subplot(2, int(math.ceil(len(lossStrList)/2.0)), index+1)
        plot_loss_face_1(iter, loss, lossStr)
        plt.tight_layout()
    plt.savefig(savePath)
    plt.close('all')

if __name__ == '__main__':
    path = '/home/sean/workplace/221/py-R-FCN-test/output/threeHusFace/Loss_VGG16_faster_rcnn_end2end_with_fuse_multianchor_frozen_v1_lha2_w2-4-1-fcn-1-1_1_b8_s1_2_m_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5.txt'
    # path = '/home/sean/workplace/221/py-R-FCN-test/output/threeHusFace/Loss_VGG16_faster_rcnn_end2end_with_fuse_multianchor_frozen_v1_lha2_w2-4-1-fcn-1-1_1_b8_s1_m_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5.txt'
    # path = '/data6/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/threeHusFace/Loss_VGG16_faster_rcnn_end2end_with_fuse_multianchor_frozen_v1_lha2_w2-4-1-fcn-1-1_1_b16_m_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5.txt'
    # path = '/data5/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/face_plus/Loss_VGG16_faster_rcnn_end2end_with_multianchor_v3-7_fc_fp3.txt'
    # path = '/data5/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/threeHusFace/Loss_VGG16_faster_rcnn_end2end_with_fuse_multianchor_frozen_v1_lha_w2-4-1-fc-3-2_fp1_2_m0.5_2_1_s2.txt'
    # path = '/data5/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/threeHusFace/Loss_VGG16_faster_rcnn_end2end_with_fuse_multianchor_frozen_v1_lha_w2-fc-1_fp1_2.txt'
    run_plot_loss(path, ['loss_keyPoint'])
    # run_plot_loss(path, ['loss_keyPoint', 'loss_bbox', 'loss_cls', 'rpn_cls_loss', 'rpn_loss_bbox', 'all_loss'])
    # run_plot_loss(path, ['loss_keyPoint', 'loss_bbox', 'loss_age', 'loss_gender', 'loss_ethnicity'])

    # import sys
    # if len(sys.argv) == 2:
    #     fn = sys.argv[1]
    #
    # else:
    #     fn = 'pycharm_log.log'
    #
    # # fn = '/home/rick/Space/clone/py-faster-rcnn/experiments/logs/faster_rcnn_end2end_VGG16_.txt.2017-04-20_10-31-36'
    # i = 0
    # while True:
    #     i+=1
    #     data = get_log_data(fn)
    #     iter, loss = get_iter_loss(data)
    #     plot_loss(iter,loss,i)
    #
    # raw_input('Enter to stop')
