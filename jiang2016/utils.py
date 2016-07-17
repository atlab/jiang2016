import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.pyplot import fill, cm, Rectangle
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay


def rasterize(node_coords, connected_pairs, region, resolution=1):
    """
    Rasterizes a neuron graph

    :param node_coords: coordinates of the single nodes
    :param connected_pairs: index pairs of connected nodes
    :param region: label vector for regions (1=DENDRITE, 2=CELLBODY, 3=AXON)
    :param resolution: raster step width along the edges
    :return: rasterized points in 3d, new region labels
    """
    X = []
    reg = []
    for fro, to in connected_pairs:
        node = node_coords[fro]
        conn = node_coords[to]
        v = (conn - node)
        l = np.sqrt((v ** 2).sum())
        if l > 0:
            v /= l
            for delta in np.arange(resolution, l, resolution):
                X.append(node + delta * v)
                reg.append(region[fro])
    return np.vstack((X, node_coords)), np.hstack((reg, region))


def plot_skeleton(ax, node_coords, skeleton, mask, mask_kw, other_kw, fast=True, stride=1):
    """
    Plots cell skeleton in given axes. Mask can be used to mark different regions. Two keyword dictionaries
    control how different regions are plotted.

    :param ax: axis handle
    :param node_coords: node locations
    :param skeleton: connectivity pairs
    :param mask: masks for region specificity (same length as node_coords)
    :param mask_kw: plotting kwargs for mask points
    :param other_kw: plotting kwargs for ~mask points
    :param fast: plot only points, not connections
    :param stride: skip every stride connection when plotting connections
    """
    if not fast:
        n = len(skeleton)
        for i, (fro, to) in enumerate(skeleton[::stride]):
            if i % 100 == 0:
                print(i, '/', n / stride)
            node = node_coords[fro]
            conn = node_coords[to]
            if mask[fro] and mask[to]:
                ax.plot(*zip(node, conn), linestyle='-', **mask_kw)
            else:
                ax.plot(*zip(node, conn), linestyle='-', **other_kw)

    else:
        ax.plot(node_coords[mask, 0], node_coords[mask, 1], node_coords[mask, 2], '.', **mask_kw)
        ax.plot(node_coords[~mask, 0], node_coords[~mask, 1], node_coords[~mask, 2], '.', **other_kw)


def spin_shuffle(X, copy=1):
    """
    Shuffles the rows of X by random rotations in the xy plane.
    :param X: data
    :param copy: multiplication factor of the data; there will be len(X)*mult datapoints in the return value
    :return: multiplied and rotation shuffled data
    """
    if copy > 1:
        X = np.vstack([np.array(X) for _ in range(copy)])
    return np.vstack([spin(x, th) for x, th in zip(X, np.random.rand(len(X)) * 2 * np.pi)])


def spin(X, theta):
    """
    Rotates the rows of X by theta.

    :param X: data matrix
    :param theta: angle in radians
    :return: rotated data
    """
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(X, R.T)


def plot_cells(X, Y, delta, param1, param2, threed=False):
    Y = Y + delta
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(12, 12), dpi=400)

    if threed:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], '.', ms=.5, **param1)
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], '.', ms=.5, **param2)
        ax.set_zlabel(r'relative cortical depth [$\mu$m]')
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(X[:, 1], X[:, 2], '.', ms=.5, **param1)
        ax.plot(Y[:, 1], Y[:, 2], '.', ms=.5, **param2)

    ax.set_xlabel(r'parallel to cut [$\mu$m]')
    ax.set_ylabel(r'perpendicular to cut [$\mu$m]')

    lgnd = ax.legend()
    for h in lgnd.legendHandles:
        h._legmarker.set_markersize(15)

    return fig, ax


def plot_projections(X, Y, delta, param1, param2):
    Y = Y + delta
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=400, sharey=True, sharex=True)

    ax_labels = dict(
        enumerate([r'x (medial-lateral) [$\mu$m]', r'y (perpendicular to cut) [$\mu$m]', r'z (depth) [$\mu$m]']))
    for k, (i, j) in enumerate(itertools.combinations([0, 1, 2], r=2)):
        ax[k].plot(X[:, i], X[:, j], '.', ms=.5, **param1)
        ax[k].plot(Y[:, i], Y[:, j], '.', ms=.5, **param2)
        ax[k].set_xlabel(ax_labels[i])
        ax[k].set_ylabel(ax_labels[j])


    ax[0].set_aspect(1)
    lgnd = ax[2].legend()
    for h in lgnd.legendHandles:
        h._legmarker.set_markersize(15)
    fig.tight_layout()
    return fig, ax


def compute_overlap_density(X, Y, bin_width, delta, n):
    """
    Computes the overlap density between X and Y shifted by dist along the x axis.

    :param X: data points for neuron 1
    :param Y: data points for neuron 2
    :param bin_width: bin width. Bins will be bin_width x bin_width x bin_width
    :param delta: amount by which Y will be shifted relative to X
    :return: overlap density, bin borders of the histograms
    """
    Y = Y + delta
    ma = np.vstack((X, Y)).max(axis=0)
    mi = np.vstack((X, Y)).min(axis=0)
    bins = tuple((np.hstack((np.arange(0, low, -bin_width)[::-1], np.arange(bin_width, high + bin_width, bin_width)))
                  for low, high in zip(mi, ma)))

    H1, _ = np.histogramdd(X, bins)
    H2, E = np.histogramdd(Y, bins)
    H1 /= bin_width ** 3 * n[0]
    H2 /= bin_width ** 3 * n[1]
    return H1 * H2, E


def extended_hinton(ax, V, C, vmax=None, cmin=None, cmax=None, cmap=None, matrix_style=False, alpha=1,
                    enforce_box=False):
    if cmap is None:
        cmap = cm.jet

    if vmax is None:  vmax = np.amax(np.abs(V))
    if cmax is None:  cmax = np.amax(C)
    if cmin is None:  cmin = np.amin(C)

    cnorm = Normalize(vmin=cmin, vmax=cmax, clip=True)
    cmapable = ScalarMappable(norm=cnorm, cmap=cmap)

    if matrix_style:
        V, C = V.T, C.T

    ax.patch.set_facecolor([0, 0, 0, 0])

    for (x, y), w in np.ndenumerate(V):
        s = C[x, y]
        color = cmap(cnorm(s))  # cmap(s / cmax)
        size = np.abs(w / vmax)
        rect = Rectangle([x - size / 2, y - size / 2], size, size,
                         facecolor=color, edgecolor=color, alpha=alpha)
        ret = ax.add_patch(rect)

    if enforce_box:
        ax.axis('tight')
        try:
            ax.set_aspect('equal', 'box')
        except:
            pass
    # ax.autoscale_view()
    # ax.invert_yaxis()
    return cnorm, cmapable


def plot_connections(P, Q, vmax=None, cmin=None, cmax=None):
    # colors = ["black", "azure","apple","golden yellow",   "neon pink"]
    # cmap = sns.blend_palette(sns.xkcd_palette(colors), as_cmap=True)
    cmap = plt.cm.get_cmap('viridis')

    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=(4.6, 3.5), dpi=400)
    gs = plt.GridSpec(10, 11)

    axes = [fig.add_subplot(gs[2:, 1:6]), fig.add_subplot(gs[2:, 6:])]
    labels = list(P.index)
    n = len(labels)
    ax_color = fig.add_subplot(gs[:2, :6])

    with sns.axes_style('ticks'):
        ax_correlation = fig.add_subplot(gs[:3, 6:])

    for p, ax in zip([P, Q], axes):
        p = p.as_matrix()
        print(p.max())
        cnorm, cmapable = extended_hinton(ax, p, p, matrix_style=True, cmap=cmap,
                                          vmax=vmax if vmax is not None else p.max(),
                                          cmin=cmin, cmax=cmax, enforce_box=True)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlim((-1, n))
        ax.set_ylim((-1, n))
        ax.set_xlabel('presynaptic', fontsize=6)

    cbar = ColorbarBase(ax_color, cmap=cmap, norm=cnorm, orientation='horizontal')

    axes[0].set_ylabel('postsynaptic', fontsize=6)
    axes[0].set_yticklabels(labels)
    axes[1].set_yticklabels([])
    for ax in axes:
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(which='major', axis='both', linewidth=.3)
        for _, i in ax.spines.items():
            i.set_linewidth(0.3)
    ax_color.tick_params(axis='both', which='major', labelsize=6)
    ax_color.set_xlabel('connection probability', fontsize=6)

    for _, i in ax_color.spines.items():
        i.set_linewidth(0)

    fig.tight_layout()

    return fig, {'matrix': axes, 'color': ax_color, 'correlation': ax_correlation}


def layer(name):
    if 'L1' in name:
        return 1
    elif 'L23' in name:
        if 'Pyr' in name:
            return 3
        else:
            return 2
    else:
        if 'Pyr' in name:
            return 5
        else:
            return 4


def load_data():
    """
    Loads Xioalong's connectivity matrix.

    :return: labels, K, N
    """
    path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    with open(path + '/files/matrix09112015.csv', 'r') as fid:
        labels = [e.strip() for e in fid.readline().split(',')[1:]]
        K, N = [], []
        for l in fid.readlines():
            K.append([list(map(float, e.strip().split('/')))[0] for e in l.split(',')[1:]])
            N.append([list(map(float, e.strip().split('/')))[1] for e in l.split(',')[1:]])
    layers = [layer(name) for name in labels]
    return labels, np.asarray(K), np.asarray(N)
