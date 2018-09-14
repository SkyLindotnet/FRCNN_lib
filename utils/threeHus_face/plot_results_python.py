import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import linecache
import numpy as np
from scipy.integrate import simps
from numpy import trapz

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def load_index_dict(str, sep = ' '):
    index_dict = {}
    content_list = str.strip().split(sep)
    for index, content in enumerate(content_list):
        index_dict[content] = index
    return index_dict

def plot_results(version, threeHusOtherMetricsDir, methods, methods_name=None, savePath=None, x_limit=0.1, colors=None, markers=None, linewidth=2,
                 fontsize=12, figure_size=(11, 6), view='all', bin_sep=0.0005):
    r"""
    Method that generates the 300W Faces In-The-Wild Challenge (300-W) results
    in the form of Cumulative Error Distributions (CED) curves. The function
    renders the indoor, outdoor and indoor + outdoor results based on both 68
    and 51 landmark points in 6 different figures.

    Please cite:
    C. Sagonas, E. Antonakos, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. "300
    Faces In-The-Wild Challenge: Database and Results", Image and Vision
    Computing, 2015.
    
    Parameters
    ----------
    version : 1 or 2
        The version of the 300W challenge to use. If 1, then the reported
        results are the ones of the first conduct of the competition in the
        ICCV workshop 2013. If 2, then the reported results are the ones of
        the second conduct of the competition in the IMAVIS Special Issue 2015.
    x_limit : float, optional
        The maximum value of the horizontal axis with the errors.
    colors : list of colors or None, optional
        The colors of the lines. If a list is provided, a value must be
        specified for each curve, thus it must have the same length as the
        number of plotted curves. If None, then the colours are linearly sampled
        from the jet colormap. Some example colour values are:

                'r', 'g', 'b', 'c', 'm', 'k', 'w', 'orange', 'pink', etc.
                or
                (3, ) ndarray with RGB values

    linewidth : float, optional
        The width of the rendered lines.
    fontsize : int, optional
        The font size that is applied on the axes and the legend.
    figure_size : (float, float) or None, optional
        The size of the figure in inches.
    """
    # Check version
    results_folder = '300W_v{}'.format(int(version))
    # MetricsDir = os.path.join(threeHusOtherMetricsDir, results_folder)
    # MetricsDirFiles = os.listdir(MetricsDir)
    # participants = filter(lambda x : x.endswith('.txt'), MetricsDirFiles)
    # participants = ['Ours_v1.txt', 'Deng.txt', 'Ours.txt', 'Fan.txt', 'Martinez.txt', 'Uricar.txt', 'Cech.txt']
    if version == 1:
        participants = ['Baltrusaitis', 'Hasan', 'Jaiswal', 'Milborrow', 'Yan',
                        'Zhou']
    elif version == 2:
        participants = ['Cech et al', 'Deng et al', 'Fan et al', 'Martinez et al', 'Uricar et al']
    else:
        raise ValueError('version must be either 1 or 2')

    participants.extend(methods)

    # Initialize lists
    ced68 = []
    ced68_indoor = []
    ced68_outdoor = []
    ced51 = []
    ced51_indoor = []
    ced51_outdoor = []
    legend_entries = []

    ced68_auc = []
    ced68_indoor_auc = []
    ced68_outdoor_auc = []
    ced51_auc = []
    ced51_indoor_auc = []
    ced51_outdoor_auc = []
    ced68_failure = []
    ced51_failure = []

    # Load results
    if view == 'all':
        for i, filename in enumerate(participants):
            # Read file
            pathStr = os.path.join(threeHusOtherMetricsDir, results_folder, filename)
            tmp = np.loadtxt(pathStr, skiprows=4)
            # Get CED values
            bins = tmp[:, 0]
            ced68.append(tmp[:, 1])
            ced68_indoor.append(tmp[:, 2])
            ced68_outdoor.append(tmp[:, 3])
            ced51.append(tmp[:, 4])
            ced51_indoor.append(tmp[:, 5])
            ced51_outdoor.append(tmp[:, 6])
            # Update legend entries
            # legend_entries.append(filename[:-4] + ' et al.')
            legend_entries.append(filename)
            # add auc into legend
            bins_r = bins / (bins[-1]-bins[0])
            ced68_auc.append(np.round(metrics.auc(bins_r, ced68[i]), 5))
            ced68_indoor_auc.append(np.round(metrics.auc(bins_r, ced68_indoor[i]), 6))
            ced68_outdoor_auc.append(np.round(metrics.auc(bins_r, ced68_outdoor[i]), 6))
            ced51_auc.append(np.round(metrics.auc(bins_r, ced51[i]), 5))
            ced51_indoor_auc.append(np.round(metrics.auc(bins_r, ced51_indoor[i]), 5))
            ced51_outdoor_auc.append(np.round(metrics.auc(bins_r, ced51_outdoor[i]), 5))


        fig = plt.figure()
        ax = plt.subplot(2, 3, 1)
        # assign legend according to auc
        ind = np.argsort(-1*np.array(ced68_auc))
        sorted_legend_entries = [legend_entries[i]+'-'+np.str(ced68_auc[i]) for i in ind]
        sorted_ced68 = np.array(ced68)[[ind]]
        # 68 points, indoor + outdoor
        title = 'Indoor + Outdoor, 68 points'
        _plot_curves(bins, sorted_ced68, sorted_legend_entries, title, x_limit=x_limit,
                     colors=colors, linewidth=linewidth, fontsize=fontsize,
                     figure_size=figure_size, ax=ax, fig=fig)
        # 68 points, indoor
        ax = plt.subplot(2, 3, 2)
        # assign legend according to auc
        ind = np.argsort(-1*np.array(ced68_indoor_auc))
        sorted_legend_entries = [legend_entries[i]+'-'+np.str(ced68_indoor_auc[i]) for i in ind]
        sorted_ced68_indoor = np.array(ced68_indoor)[[ind]]
        title = ''
        _plot_curves(bins, sorted_ced68_indoor, sorted_legend_entries, title, x_limit=x_limit,
                     colors=colors, linewidth=linewidth, fontsize=fontsize,
                     figure_size=figure_size, ax=ax, fig=fig)
        # 68 points, outdoor
        ax = plt.subplot(2, 3, 3)
        # assign legend according to auc
        ind = np.argsort(-1*np.array(ced68_outdoor_auc))
        sorted_legend_entries = [legend_entries[i]+'-'+np.str(ced68_outdoor_auc[i]) for i in ind]
        sorted_ced68_outdoor = np.array(ced68_outdoor)[[ind]]
        title = ''
        _plot_curves(bins, sorted_ced68_outdoor, sorted_legend_entries, title, x_limit=x_limit,
                     colors=colors, linewidth=linewidth, fontsize=fontsize,
                     figure_size=figure_size, ax=ax, fig=fig)
        # 51 points, indoor + outdoor
        ax = plt.subplot(2, 3, 4)
        # assign legend according to auc
        ind = np.argsort(-1*np.array(ced51_auc))
        sorted_legend_entries = [legend_entries[i]+'-'+np.str(ced51_auc[i]) for i in ind]
        sorted_ced51 = np.array(ced51)[[ind]]
        title = 'Indoor + Outdoor, 51 points'
        _plot_curves(bins, sorted_ced51, sorted_legend_entries, title, x_limit=x_limit,
                     colors=colors, linewidth=linewidth, fontsize=fontsize,
                     figure_size=figure_size, ax=ax, fig=fig)
        # 51 points, indoor
        ax = plt.subplot(2, 3, 5)
        # assign legend according to auc
        ind = np.argsort(-1*np.array(ced51_indoor_auc))
        sorted_legend_entries = [legend_entries[i]+'-'+np.str(ced51_indoor_auc[i]) for i in ind]
        sorted_ced51_indoor = np.array(ced51_indoor)[[ind]]
        title = 'Indoor, 51 points'
        _plot_curves(bins, sorted_ced51_indoor, sorted_legend_entries, title, x_limit=x_limit,
                     colors=colors, linewidth=linewidth, fontsize=fontsize,
                     figure_size=figure_size, ax=ax, fig=fig)
        # 51 points, outdoor
        ax = plt.subplot(2, 3, 6)
        # assign legend according to auc
        ind = np.argsort(-1*np.array(ced51_outdoor_auc))
        sorted_legend_entries = [legend_entries[i]+'-'+np.str(ced51_outdoor_auc[i]) for i in ind]
        sorted_ced51_outdoor = np.array(ced51_outdoor)[[ind]]
        title = 'Outdoor, 51 points'
        _plot_curves(bins, sorted_ced51_outdoor, sorted_legend_entries, title, x_limit=x_limit,
                     colors=colors, linewidth=linewidth, fontsize=fontsize,
                     figure_size=figure_size, ax=ax, fig=fig)

        plt.tight_layout()
        plt.savefig(savePath, dpi=300)
        plt.close('all')
    elif view == 'lite':
        for i, filename in enumerate(participants):
            # Read file
            pathStr = os.path.join(threeHusOtherMetricsDir, results_folder, filename)
            index_dict = load_index_dict(linecache.getline(pathStr + '.txt', 4))
            tmp = np.loadtxt(pathStr + '.txt', skiprows=4)
            # Get CED values
            bins = tmp[:, index_dict['Bin']]  # 0
            ced68.append(tmp[:, index_dict['68_all']])  # 1 1
            ced51.append(tmp[:, index_dict['51_all']])  # 4 2
            # Update legend entries
            # legend_entries.append(filename[:-4] + ' et al.')
            if filename in methods:
                legend_entries.append(methods_name[methods.index(filename)])
            else:
                legend_entries.append(filename)

            # another rule 2013
            # simps(ced68[i][0:161:20], dx=0.01) / 0.08

            # add auc into legend 2015
            max_index = x_limit / bin_sep
            auc68 = simps(ced68[i][0:max_index+1], dx=bin_sep) / x_limit
            auc51 = simps(ced51[i][0:max_index+1], dx=bin_sep) / x_limit

            ced68_auc.append(np.round(auc68, 4))
            ced51_auc.append(np.round(auc51, 4))

            # add failure rate
            ced68_failure.append(np.round(1 - ced68[i][max_index], 4))
            ced51_failure.append(np.round(1 - ced51[i][max_index], 4))

        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        # assign legend according to auc
        ind = np.argsort(-1 * np.array(ced68_auc))
        # sorted_legend_entries = [legend_entries[i] + '-' + np.str(ced68_auc[i]) +
        #                          '-%s' % np.str(ced68_failure[i]) for i in ind]
        sorted_legend_entries = [legend_entries[i] for i in ind]
        sorted_ced68 = np.array(ced68)[[ind]]
        # 68 points, indoor + outdoor
        title = 'Indoor + Outdoor, 68 points'
        title = ''
        _plot_curves(bins, sorted_ced68, sorted_legend_entries, title, x_limit=x_limit,
                     colors=colors, linewidth=linewidth, fontsize=fontsize,
                     figure_size=figure_size, ax=ax, fig=fig)
        # 51 points, indoor + outdoor
        ax = plt.subplot(1, 2, 2)
        # assign legend according to auc
        ind = np.argsort(-1 * np.array(ced51_auc))
        # sorted_legend_entries = [legend_entries[i] + '-' + np.str(ced51_auc[i]) +
        #                          '-%s' % np.str(ced51_failure[i]) for i in ind]
        sorted_legend_entries = [legend_entries[i] for i in ind]
        sorted_ced51 = np.array(ced51)[[ind]]
        title = 'Indoor + Outdoor, 51 points'
        title = ''
        _plot_curves(bins, sorted_ced51, sorted_legend_entries, title, x_limit=x_limit,
                     colors=colors, linewidth=linewidth, fontsize=fontsize,
                     figure_size=figure_size, ax=ax, fig=fig)

        plt.tight_layout()
        plt.savefig(savePath, dpi=300)
        plt.close('all')


def plot_results_kp(MetricsDir, savePath=None, x_limit=1, colors=None, markers=None, linewidth=3,
                 fontsize=12, figure_size=(4.8, 4)):
    # Check version
    MetricsDirFiles = os.listdir(MetricsDir)
    participants = filter(lambda x: x.endswith('.txt'), MetricsDirFiles)
    # Initialize lists
    ced68 = []
    legend_entries = []
    ced68_auc = []
    bins = []

    # Load results
    for i, filename in enumerate(participants):
        # Read file
        pathStr = os.path.join(MetricsDir, filename)
        tmp = np.loadtxt(pathStr, skiprows=4)
        # Get CED values
        bins.append(tmp[:, 0])
        ced68.append(tmp[:, 1])
        # Update legend entries
        legend_entries.append(filename[:-4])
        # add auc into legend
        bins_r = bins[i] / (bins[i][-1]-bins[i][0])
        ced68_auc.append(np.round(metrics.auc(bins_r, ced68[i]), 5))

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    # assign legend according to auc
    ind = np.argsort(-1*np.array(ced68_auc))
    sorted_legend_entries = [legend_entries[i]+'-'+np.str(ced68_auc[i]) for i in ind]
    sorted_ced68 = np.array(ced68)[[ind]]
    sorted_bins = np.array(bins)[[ind]]
    # 68 points, indoor + outdoor
    title = ''
    _plot_curves_gen(sorted_bins, sorted_ced68, sorted_legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size, ax=ax, fig=fig, xlabel='Point-to-point Normalized RMS Error',
                 ylabel='Valid Faces Proportion')
    plt.tight_layout()
    plt.savefig(savePath)
    plt.close('all')


def plot_results_det(MetricsDir, savePath=None, x_limit=1, colors=None, markers=None, linewidth=3,
                 fontsize=12, figure_size=(4.8, 4)):
    # Check version
    MetricsDirFiles = os.listdir(MetricsDir)
    participants = filter(lambda x: x.endswith('.txt'), MetricsDirFiles)
    # Initialize lists
    precs = []
    recs = []
    legend_entries = []
    aps = []

    # Load results
    for i, filename in enumerate(participants):
        # Read file
        pathStr = os.path.join(MetricsDir, filename)
        tmp = np.loadtxt(pathStr, skiprows=2)
        # Get CED values
        recs.append(tmp[:, 0])
        precs.append(tmp[:, 1])
        # Update legend entries
        legend_entries.append(filename[:-4])
        # add ap into legend
        with open(pathStr, 'r') as f:
            ap = np.float(f.readline())
        aps.append(np.round(ap, 5))

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    # assign legend according to auc
    ind = np.argsort(-1*np.array(aps))
    sorted_legend_entries = [legend_entries[i]+'-'+np.str(aps[i]) for i in ind]
    sorted_precs = np.array(precs)[[ind]]
    sorted_recs = np.array(recs)[[ind]]
    title = ''
    _plot_curves_gen(sorted_recs, sorted_precs, sorted_legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size, ax=ax, fig=fig, xlabel='Recall',
                 ylabel='Precision')
    plt.tight_layout()
    plt.savefig(savePath)
    plt.close('all')


def _load_metric_info(pathStr, targetStr='det_AP', iter=10000):
    with open(pathStr, 'r') as f:
        lines = f.readlines()
    metric_iters = np.array((np.arange(len(lines)) + 1) * iter, dtype=np.float)
    metric_targets = []
    for line in lines:
        split_line = line[:-1].split(' ')[1:]
        target_str = filter(lambda k: k.find(targetStr) != -1, split_line)[0]
        target = target_str.split(':')[1]
        metric_targets.append(target)
    assert len(metric_iters) == len(metric_targets)
    return metric_iters, np.array(metric_targets, dtype=np.float)


def plot_results_metric(MetricsDir, targetStr='det_AP', savePath=None, x_limit=1, colors=None, markers=None, linewidth=3,
                 fontsize=12, figure_size=(4.8, 4)):
    # Check version
    MetricsDirFiles = os.listdir(MetricsDir)
    participants = filter(lambda x: x.endswith('.txt'), MetricsDirFiles)
    # Initialize lists
    y = []
    legend_entries = []
    ced68_auc = []
    x = []

    participants.sort()
    participants.reverse()

    # Load results
    for i, filename in enumerate(participants):
        # Read file
        pathStr = os.path.join(MetricsDir, filename)

        iters, targets = _load_metric_info(pathStr, targetStr)
        # Get CED values
        x.append(iters)
        y.append(targets)
        # Update legend entries
        legend_entries.append(filename[:-4])
        # add auc into legend
        bins_r = x[i] / (x[i][-1]-x[i][0])
        ced68_auc.append(np.round(metrics.auc(bins_r, y[i]), 5))

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    # assign legend according to auc
    ind = np.argsort(-1*np.array(ced68_auc))
    ind = np.arange(len(ind))
    # sorted_legend_entries = [legend_entries[i]+'-'+np.str(ced68_auc[i]) for i in ind]
    sorted_legend_entries = [legend_entries[i] for i in ind]
    sorted_ced68 = np.array(y)[[ind]]
    sorted_bins = np.array(x)[[ind]]
    # 68 points, indoor + outdoor
    title = ''
    if targetStr == 'det_AP':
        targetStr = 'detection AP'
    _plot_polyline_gen(sorted_bins/10000, sorted_ced68, sorted_legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size, ax=ax, fig=fig, xlabel='Iterations(x 10,000)',
                 ylabel=targetStr)

    plt.tight_layout()
    plt.savefig(savePath)
    plt.close('all')


def _plot_polyline_gen(bins, ced_values, legend_entries, title, x_limit=0.08,
                 colors=None, linewidth=3, fontsize=12, figure_size=None, ax=None, fig=None,
                 xlabel='xlabel', ylabel='ylabel'):
    # number of curves
    n_curves = len(ced_values)

    # if no colors are provided, sample them from the jet colormap
    if colors is None:
        cm = plt.get_cmap('jet')
        colors = [cm(1.*i/n_curves)[:3] for i in range(n_curves)]

    # plot all curves
    # fig = plt.figure()
    # ax = plt.gca
    markers = ['o', '*', 'v', '^']
    if len(bins) == 1:
        for i, y in enumerate(ced_values):
            plt.plot(bins[0], y, color=colors[i],
                     linestyle='-',
                     linewidth=linewidth,
                     label=legend_entries[i])
    else:
        for i, y in enumerate(ced_values):
            plt.plot(bins[i], y, color=colors[i],
                     marker='o', mec='r', mfc='w',
                     linestyle='-',
                     linewidth=linewidth,
                     label=legend_entries[i])

    # legend
    ax.legend(prop={'size': fontsize}, loc=0)

    # axes
    for l in (ax.get_xticklabels() + ax.get_yticklabels()):
        l.set_fontsize(fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # set axes limits
    # ax.set_xlim([0., x_limit])
    # ax.set_ylim([0., 1.])
    # ax.set_yticks(np.arange(0., 1.2, 0.2))
    # ax.set_xticks(np.arange(0., x_limit + 0.2))

    # grid
    plt.grid('on', linestyle='--', linewidth=0.5)

    # figure size
    if figure_size is not None:
        fig.set_size_inches(np.asarray(figure_size))


def _plot_curves_gen(bins, ced_values, legend_entries, title, x_limit=0.08,
                 colors=None, linewidth=3, fontsize=12, figure_size=None, ax=None, fig=None,
                 xlabel='xlabel', ylabel='ylabel'):
    # number of curves
    n_curves = len(ced_values)

    # if no colors are provided, sample them from the jet colormap
    if colors is None:
        cm = plt.get_cmap('jet')
        colors = [cm(1.*i/n_curves)[:3] for i in range(n_curves)]

    # plot all curves
    # fig = plt.figure()
    # ax = plt.gca()
    if len(bins) == 1:
        for i, y in enumerate(ced_values):
            plt.plot(bins[0], y, color=colors[i],
                     linestyle='-',
                     linewidth=linewidth,
                     label=legend_entries[i])
    else:
        for i, y in enumerate(ced_values):
            plt.plot(bins[i], y, color=colors[i],
                     linestyle='-',
                     linewidth=linewidth,
                     label=legend_entries[i])

    # legend
    ax.legend(prop={'size': fontsize}, loc=0)

    # axes
    for l in (ax.get_xticklabels() + ax.get_yticklabels()):
        l.set_fontsize(fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # set axes limits
    ax.set_xlim([0., x_limit])
    ax.set_ylim([0., 1.])
    ax.set_yticks(np.arange(0., 1.1, 0.1))
    ax.set_xticks(np.arange(0., x_limit + 0.2, 0.2))

    # grid
    plt.grid('on', linestyle='--', linewidth=0.5)

    # figure size
    if figure_size is not None:
        fig.set_size_inches(np.asarray(figure_size))


def _plot_curves(bins, ced_values, legend_entries, title, x_limit=0.08,
                 colors=None, linewidth=3, fontsize=12, figure_size=None, ax=None, fig=None):
    # number of curves
    n_curves = len(ced_values)
    
    # if no colors are provided, sample them from the jet colormap
    if colors is None:
        cm = plt.get_cmap('jet')
        colors = [cm(1.*i/n_curves)[:3] for i in range(n_curves)]
        
    # plot all curves
    # fig = plt.figure()
    # ax = plt.gca()
    for i, y in enumerate(ced_values):
        plt.plot(bins, y, color=colors[i],
                 linestyle='-',
                 linewidth=linewidth, 
                 label=legend_entries[i])
        
    # legend
    ax.legend(prop={'size': fontsize}, loc=0)
    
    # axes
    for l in (ax.get_xticklabels() + ax.get_yticklabels()):
        l.set_fontsize(fontsize)
    ax.set_xlabel('Point-to-point Normalized RMS Error', fontsize=fontsize)
    ax.set_ylabel('Images Proportion', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # set axes limits
    ax.set_xlim([0., x_limit])
    ax.set_ylim([0., 1.])
    ax.set_yticks(np.arange(0., 1.1, 0.1))
    
    # grid
    plt.grid('on', linestyle='--', linewidth=0.5)
    
    # figure size
    if figure_size is not None:
        fig.set_size_inches(np.asarray(figure_size))


def _plot_intensity_scatter(component, threshold, re_nms, size, max_x=50, max_y=1.,
                            figure_size=None, fontsize=12):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.scatter(component, threshold, c=re_nms, alpha=0.5, s=size,
                cmap=plt.cm.jet)
    v = np.linspace(min(re_nms), max(re_nms), 7, endpoint=True)
    v = np.around(v, decimals=4)
    cbar = plt.colorbar(ticks=v)  # show color scale
    cbar.ax.tick_params(labelsize=fontsize)
    print fontsize

    # axes
    for l in (ax.get_xticklabels() + ax.get_yticklabels()):
        l.set_fontsize(fontsize)
    ax.set_xlabel('Active Component', fontsize=fontsize)
    ax.set_ylabel('Confidence Threshold', fontsize=fontsize+2)

    # set axes limits
    ax.set_xlim([0., max_x])
    ax.set_ylim([0., max_y])
    # ax.set_yticks(np.arange(0, 1.00, 0.05))
    # ax.set_xticks(np.arange(2, 46, 2))

    # figure size
    if figure_size is not None:
        fig.set_size_inches(np.asarray(figure_size))
    pass


def plot_psr_result(psr_path, init_error, error_cover=5e-6, size_focus=0,
                    fontsize=12, figure_size=(7, 6)):
    tmp = np.loadtxt(psr_path, skiprows=2)
    component = tmp[:, 0]
    threshold = tmp[:, 2]
    re_nms = tmp[:, 4] * 100
    max_x = max(component)+1

    index = np.where(re_nms <= init_error*100)[0]
    component = component[index]
    threshold = threshold[index]
    re_nms = re_nms[index]
    min_re_nms = min(re_nms)
    valid_index = np.where(re_nms <= min_re_nms+error_cover*100)[0]
    print 'suggested component: %d; threshold: %f' % (np.median(component[valid_index]), np.median(threshold[valid_index]))
    # re_nms[valid_index] = [0.021785]*len(valid_index)
    size = np.array([50]*len(re_nms))
    if size_focus:
        size[valid_index] = [100]*len(valid_index)
    _plot_intensity_scatter(component, threshold, re_nms, size, max_x=max_x, max_y=1.,
                            fontsize=fontsize, figure_size=figure_size)
    pass

if __name__ == '__main__':
    psr_path = '/home/sean/workplace/221/py-R-FCN-test/output/test_result/threeHusFace/Det_IMG/threeHusFace+/IMG_VGG16_FMA_frozen_v1_lha5-re-1(fwc)_w2-4-1-fcn-5-1_1_b4_s1_2_fm_p345(14-11-9)-c-norm_fuse_4_dh-4(d)-gpu2-i_dc3_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5_iter_160000_c1_n3_adapted_gt(fwc-1)_train/PSR_record_all.txt'
    plot_psr_result(psr_path, init_error=0.021585, error_cover=3e-6, size_focus=0)  # 0.021785

    # psr_path = '/home/sean/workplace/221/py-R-FCN-test/output/test_result/cofw/Det_IMG/cofw/IMG_VGG16_FMA_frozen_v1_cofw-1(fwc-2)_w2-4-1-fcn-5-1_1_b4_s1_2_fm_p345(14-11-9)-c-norm_fuse_4_dh-4(d)-gpu2_dc3_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5_iter_160000_c1_n3_adapted_gt(fwc-2)_train/PSR_record_all.txt'
    # plot_psr_result(psr_path, init_error=0.019460, error_cover=3e-6, size_focus=0)  # 0.019450

    # psr_path = '/home/sean/workplace/221/py-R-FCN-test/output/test_result/aflw-pifa/Det_IMG/aflw-pifa/IMG_VGG16_FMA_frozen_v1_aflw-pifa-1(fwc-1)_w2-4-1-fcn-5-1_1_b4_s1_2_fm_p345(14-11-9)-c-norm_fuse_4_dh-4(d)-gpu3_dc3_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5_iter_160000_c1_n3_adapted_gt(fwc-1)_train/PSR_record_all_1.txt'
    # plot_psr_result(psr_path, init_error=0.026633, error_cover=3e-6, size_focus=0)  # 0.024533
    exit(1)

    # plot_results(2)
    dir = '/home/sean/workplace/221/py-R-FCN-test/data/DB/face/300-w_face/300W_results/'
    # methods = ['Metric_VGG16_FMA_frozen_v1_lha5-re-1(fwc)_w2-4-1-fcn-5-1_1_b4_s1_2_fm_p345(14-11-9)-c-norm_fuse_4_dh-4(d)-gpu2-i_dc3_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5_iter_160000_c1_n3_adapted_gt(fwc-1)_300w-v1',
    #            'Metric_VGG16_FMA_frozen_v1_lha5-re-1(fwc)_w2-4-1-fcn-5-1_1_b4_s1_2_fm_p345(14-11-9)-c-norm_fuse_4_dh-4(d)-gpu2-i_dc3_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5_iter_160000_c1_n3_unscale_m1000_adapted_gt(fwc-1)_300w-v1',
    #            'Metric_VGG16_FMA_frozen_v1_lha5-re-1(fwc)_w2-4-1-fcn-5-1_1_b4_s1_2_fm_p345(14-11-9)-c-norm_fuse_4_dh-4(d)-gpu2-i_dc3_fp3_v3-10_m0.5_2_1_s2-roi-norm-10-8-5_iter_160000_c1_n3_unscale_m1800_adapted_gt(fwc-1)_300w-v1']
    methods = ['RCEN', 'RCEN+PSR1']
    # methods_name = ['Ours', 'Ours-unscale-m1000', 'Ours-unscale-m1800']
    methods_name = ['RCEN', 'RCEN+PSR']  # , 'Ours-pca'
    plot_results(2, dir, methods, methods_name, view='lite', linewidth=4, fontsize=16)
    # dir = '/data6/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/test_result/face_plus/Metric/Face_Plus_v1/compare'
    # plot_results_kp(dir)

    # metricdir = '/data6/yyliang/cuda-workspace/py-faster-rcnn_face/py-R-FCN-test/output/face_plus/compare'
    # plot_results_metric(metricdir, targetStr='det_AP')
    # plot_results_metric(metricdir, targetStr='det_ETHNICITY_ACC')