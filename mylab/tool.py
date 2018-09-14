import numpy as np
from easydict import EasyDict as edict
import time
import sys, os, shutil

def median(midlist):
    midlist.sort()
    lens = len(midlist)
    if lens % 2 != 0:
        midl = (lens / 2)
        res = midlist[midl]
    else:
        odd = (lens / 2) - 1
        ev = (lens / 2)
        res = float(midlist[odd] + midlist[ev]) / float(2)
    return res

class ProgressBar(object):
    DEFAULT_BAR_LENGTH = 65
    DEFAULT_CHAR_ON = '='
    DEFAULT_CHAR_OFF = ' '

    def __init__(self, end, start=0):
        self.end = end
        self.start = start
        self._barLength = self.__class__.DEFAULT_BAR_LENGTH
        self.setLevel(self.start)
        self._plotted = False

    def setLevel(self, level):
        self._level = level
        if level < self.start:  self._level = self.start
        if level > self.end:    self._level = self.end

        self._ratio = float(self._level - self.start) / float(self.end - self.start)
        self._levelChars = int(self._ratio * self._barLength)

    def plotProgress(self):
        sys.stdout.write("\r  %3i%% [%s%s]" % (
            int(self._ratio * 100.0),
            self.__class__.DEFAULT_CHAR_ON * int(self._levelChars),
            self.__class__.DEFAULT_CHAR_OFF * int(self._barLength - self._levelChars),
        ))
        sys.stdout.flush()
        self._plotted = True

    def plotProgress2(self, s):
        sys.stdout.write("\r  %3i%% [%s%s] %s" % (
            int(self._ratio * 100.0),
            self.__class__.DEFAULT_CHAR_ON * int(self._levelChars),
            self.__class__.DEFAULT_CHAR_OFF * int(self._barLength - self._levelChars), s))
        sys.stdout.flush()

    def setAndPlot(self, level):
        oldChars = self._levelChars
        self.setLevel(level)
        if (not self._plotted) or (oldChars != self._levelChars):
            self.plotProgress()

    def __add__(self, other):
        assert type(other) in [float, int], "can only add a number"
        self.setAndPlot(self._level + other)
        return self

    def __sub__(self, other):
        return self.__add__(-other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__add__(-other)

    def print_str(self, s):
        self.plotProgress2(s)

    def __del__(self):
        sys.stdout.write("\n")

def to_float(func):
    def func_ret(*v, **vs):
        r = func(*v, **vs)
        r= np.array(r).astype(np.float32)
        return r
    return func_ret


def timeit(func):
    def func_ret(*args, **kwargs):
        st = time.time()
        '''-----------'''
        r = func(*args, **kwargs)
        '''-----------'''
        et = time.time()
        elps = et -st
        m, s = divmod(elps, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        m = int(m)
        h = int(h)
        d = int(d)
        s = ' {:.3f}s'.format(s)
        m = ' {:d}m'.format(m) if m > 0 else ''
        h = ' {:d}h'.format(h) if h > 0 else ''
        d = ' {:d}day{}'.format(d, 's' if d > 1 else '') if d > 0 else ''
        print '@timeit({}):{}{}{}{}'.format(func.__name__, d, h, m, s)
        return r

    return func_ret


def save_dict(name, dictionary):
    '''

    :param name: full path to save the dict (suggested extension: npy)
    :param dictionary: a dict variable
    :return:
    '''
    np.save(name, dictionary)


def load_dict(name):
    '''

    :param name: full path to a saved dict
    :return: loaded dict
    '''
    return np.load(name).item()

def mk_dir(saveDir, cover=1):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    elif cover:
        shutil.rmtree(saveDir)
        os.makedirs(saveDir)

def parse_file(f, full_path=False):
    '''
    
    :param f: file_path
    :param full_path: set True to disable path prune
    :return: 
    1. ordered key of dict
        
    2. dict, key is $1 returned, value is also a dict
    in each dict, it has keys:
        'name': path (effected by arg 'full_path');
        'num' : number of faces
        'coords': a list of string
        
    '''

    def deal_file_path(s, full_path):
        if full_path:
            return s
        else:
            s = s.strip('.jpg')
            return '/'.join(s.split('/')[-5:])

    with open(f, 'r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]

    at = 0
    ret = edict()
    keys = []
    while (at < len(lines) - 1):
        d = edict()
        this = lines[at]
        name = deal_file_path(this, full_path)
        keys.append(name)
        at += 1

        this = int(lines[at])
        d['num'] = this;
        at += 1

        coords = []
        for _ in range(this):
            coords.append(lines[at]);
            at += 1

        d['coords'] = coords

        ret[name] = d

    return keys, ret


def write_out(keys, d, path):
    s = []
    for k in keys:
        s.append(k)
        d_i = d[k]
        s.append(str(d_i['num']))
        s.extend(d_i['coords'])
    s = '\n'.join(s) + '\n'

    with open(path, 'w') as f:
        f.write(s)
