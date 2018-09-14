import numpy as np
from pylab import *
from scipy.spatial import distance
from scipy.interpolate import interp1d
from scipy.io import loadmat

eps = 1e-5

def get_linear_line(p1, p2):
    # input 2 points,
    # return 1 line
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    if x1 == x2:
        return np.nan, x1

    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1

    return k, b


def point_away_from_point(p, p2, away, k, b):
    # @original point p
    # @constraint point p2 (used to make sure ret_pt_x in [p_x, p2_x])
    # @away amount
    # @a linear function lfunc

    def inbetween(a1, a, a2):
        return (a >= a1 and a <= a2) or (a >= a2 and a <= a1)

    if k is np.nan:
        y_ = p[1] + np.sign(p2[1] - p[1]) * away
        return np.array([p[0], y_])
    if k == 0:
        x_ = p[0] + np.sign(p2[0]) * away
        return np.array([x_, p[1]])

    tmp = np.sqrt((away ** 2) / float(1 + k * k))
    x_delta = tmp

    candid_x1 = p[0] + x_delta
    candid_y1 = k * candid_x1 + b

    candid_x2 = p[0] - x_delta
    candid_y2 = k * candid_x2 + b

    if inbetween(p[0], candid_x1, p2[0]) and inbetween(p[1], candid_y1, p2[1]):
        return np.array([candid_x1, candid_y1])
    elif inbetween(p[0], candid_x2, p2[0]) and inbetween(p[1], candid_y2, p2[1]):
        return np.array([candid_x2, candid_y2])

    else:
        raise Exception('???')


def seg_line(x, y, seg_num):
    # feed a line with x and y array,
    #   assume the points will not jump over anyother points
    # specify the number of segments
    # return seg_num+1 point as the cut point
    assert len(x) == len(y)

    kLines = []
    bLines = []
    distan = []

    for i in range(len(x) - 1):
        p1 = np.array([x[i], y[i]])
        p2 = np.array([x[i + 1], y[i + 1]])
        k, b = get_linear_line(p1, p2)
        kLines.append(k)
        bLines.append(b)
        distan.append(distance.euclidean(p1, p2))

    seg_len = np.sum(distan) / float(seg_num)
    ret = np.zeros(shape=(seg_num + 1, 2))
    ret[0, :] = [x[0], y[0]]
    ret[-1, :] = [x[-1], y[-1]]

    l_idx = 0
    p_idx = 0
    seg_idx = 1
    target_len = seg_len

    local_pt = [x[p_idx], y[p_idx]]
    while 1:
        if seg_idx > ret.shape[0] - 2:
            break
        remote_pt = [x[p_idx + 1], y[p_idx + 1]]
        available_distance = distance.euclidean(remote_pt, local_pt)
        if target_len > available_distance + eps:
            target_len -= available_distance
            l_idx += 1
            p_idx += 1
            local_pt = [x[p_idx], y[p_idx]]
        else:
            if abs(target_len - available_distance) <= eps: #prevent numerical problem
                T = remote_pt
            else:
                T = point_away_from_point(local_pt, remote_pt, target_len, kLines[l_idx], bLines[l_idx])
            ret[seg_idx, :] = T
            local_pt = T
            target_len = seg_len
            seg_idx += 1

    return ret


def __deal_contours(pts):
    chin_point = pts[0, :]
    left_contour = pts[1:10, :]
    left_contour = np.vstack((left_contour, chin_point))
    left_contour = seg_line(left_contour[:, 0], left_contour[:, 1], 8)

    right_contour = pts[10:19, :]
    right_contour = np.vstack((chin_point, right_contour[::-1, :]))
    right_contour = seg_line(right_contour[:, 0], right_contour[:, 1], 8)

    assert (left_contour[-1, :] == right_contour[0, :]).all(), \
        'left contour [:,-1]: {}\nright contour [:,0]: {}' \
            .format(left_contour[-1, :], right_contour[0, :])

    right_contour = right_contour[1:, :]
    ret = np.vstack((left_contour, right_contour))
    return ret


def __deal_eyebrow(pts):
    # left eyebrow
    tmp = np.zeros(shape=(2, 3, 2))
    thick_area_ind = [30, 31, 32, 34, 35, 36]
    tmp[0, ...] = pts[thick_area_ind[:3], :]
    tmp[1, ...] = pts[thick_area_ind[3:], :]
    thick_ares_mean = tmp.mean(axis=0)
    left_eyebrow = np.vstack((pts[29, :], thick_ares_mean, pts[33, :]))

    # right_eyebrow
    tmp = np.zeros(shape=(2, 3, 2))
    thick_area_ind = [48, 49, 50, 52, 53, 54]
    tmp[0, ...] = pts[thick_area_ind[:3], :]
    tmp[1, ...] = pts[thick_area_ind[3:], :]
    thick_ares_mean = tmp.mean(axis=0)
    right_eyebrow = np.vstack((pts[47, :], thick_ares_mean, pts[51, :]))

    return np.vstack((left_eyebrow, right_eyebrow))


def __deal_nose(pts):
    ind = [73, 77]
    uppers = pts[ind, :].mean(axis=0)
    lower = pts[82, :]
    nose_spine = seg_line([uppers[0], lower[0]], [uppers[1], lower[1]], 3)
    low_bow = pts[[80, 75, 76, 79, 81], :]

    return np.vstack((nose_spine, low_bow))


def __deal_eyes(pts):
    ret = np.zeros(shape=(12, 2))

    # left eye
    upper_ind = [21, 27, 26, 28, 25]
    lower_ind = [21, 22, 19, 23, 25]
    upper_line = pts[upper_ind, :]
    lower_line = pts[lower_ind, :]

    upper_line = seg_line(upper_line[:, 0], upper_line[:, 1], 3)
    lower_line = seg_line(lower_line[:, 0], lower_line[:, 1], 3)

    assert (upper_line[0, :] == lower_line[0, :]).all()
    assert (upper_line[-1, :] == lower_line[-1, :]).all()

    ret[0:4, :] = upper_line
    ret[[5, 4], :] = lower_line[[1, 2], :]

    # right eye
    upper_ind = [39, 45, 44, 46, 43]
    lower_ind = [39, 40, 37, 41, 43]
    upper_line = pts[upper_ind, :]
    lower_line = pts[lower_ind, :]

    upper_line = seg_line(upper_line[:, 0], upper_line[:, 1], 3)
    lower_line = seg_line(lower_line[:, 0], lower_line[:, 1], 3)

    assert (upper_line[0, :] == lower_line[0, :]).all()
    assert (upper_line[-1, :] == lower_line[-1, :]).all()

    ret[6:10, :] = upper_line
    ret[[11, 10], :] = lower_line[[1, 2], :]

    return ret


def __deal_mouse(pts):
    ret = np.zeros(shape=(20, 2))

    lvl1 = pts[[55, 67, 66, 72, 69, 70, 64], :]
    lvl2 = pts[[55, 68, 65, 71, 64], :]
    lvl3 = pts[[57,63,60], :]
    lvl4 = pts[[55, 58, 59, 56, 62, 61, 64], :]

    lvl2 = seg_line(lvl2[:, 0], lvl2[:, 1], 6)
    # lvl3 = seg_line(lvl3[:, 0], lvl3[:, 1], 6)

    ret[0:7, :] = lvl1
    ret[7:12, :] = lvl4[1:-1, :][::-1,:]
    ret[12:17, :] = lvl2[1:-1, :]
    ret[17:20, :] = lvl3[::-1,:]

    return ret


def landmark_reduce(pts):
    contour = __deal_contours(pts)
    eyebrow = __deal_eyebrow(pts)
    nose = __deal_nose(pts)
    eyes = __deal_eyes(pts)
    mouse = __deal_mouse(pts)
    return np.vstack([contour, eyebrow, nose, eyes, mouse])


def facepp83_to_zyx83(facepp_pts):
    #83 landmark order convert
    m = np.zeros(shape=(84,),dtype=int) #add 1 for matlab one start index
    m[1:10] = [x for x in range(1,10)] #1 ot 9
    m[10]=0 #10
    m[11:20] = [x for x in range(10,19)[::-1]]#11 to 19
    m[20]  = 29
    m[21:25] = [34,35,36,33] #21,22,23,24
    m[25:28] = [32,31,30] #25,26,27
    m[28:36] = [47,52,53,54,51,50,49,48] # 28 to 35
    m[36:46] = [21,27,26,28,25,23,19,22,24,20] #36 to 45
    m[46:56] = [39,45,44,46,43,41,37,40,42,38] #46 to 55
    m[56:66] = [73,74,80,75,76,79,81,78,77,82] #56 to 66
    m[66:78] = [55,67,66,72,69,70,64,61,62,56,59,58] #66 to 77
    m[78:84] = [68,65,71,60,63,57]
    for i in range(83):
        assert i in m, 'missing {}'.format(i)

    ret = np.zeros(shape=(83,2))
    for i in range(ret.shape[0]):

        ret[i,:] = facepp_pts[m[i+1]]
    return ret
if __name__ == '__main__':
    mat =0
