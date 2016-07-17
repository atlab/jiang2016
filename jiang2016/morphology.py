import datajoint as dj
from distutils.version import StrictVersion

assert StrictVersion(dj.__version__) >= StrictVersion('0.2.8')

import numpy as np
import matplotlib.pyplot as plt
from scipy import interp

from jiang2016.utils import rasterize, plot_cells, spin_shuffle, compute_overlap_density, plot_skeleton
from . import connectivity as conn
from itertools import product
import seaborn as sns
import pandas as pd
from .utils import extended_hinton, plot_connections, load_data, plot_projections
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

schema = dj.schema('share_jiang2016_morphology', locals())
schema.spawn_missing_classes()


def load_cells(trees, shrinkage_correction=1, resolution=.5, correct_cut=False, stack=True):
    """
    Loads cells from the database
    :param cell_type: string containing layer and cell type morphology
    :param resolution: resolution for rasterization (default = .5 mu)
    :return: rasterized 3d points, region labels
    """
    cells = []
    regions = []
    pairs = []
    CELLBODY = (CellRegion() & dict(cell_region_name='cellbody')).fetch1['cell_region_id']
    print('Loading cells with shrinkage_correction', shrinkage_correction, '\tresolution', resolution,
          '\tcut correction', correct_cut)
    for k, node_coords, connected_pairs, region in \
            zip(*trees.fetch[dj.key, 'node_coords', 'connected_pairs', 'node_region']):

        # swap y and z axis
        node_coords = np.array(node_coords)
        node_coords[:, [1, 2]] = node_coords[:, [2, 1]]

        connected_pairs = connected_pairs.astype(int) - 1  # matlab to python index shift
        region = region.squeeze()
        if not np.any(region == CELLBODY):
            print(k, 'has no cellbody')
            continue
        mu = np.mean(node_coords[region == CELLBODY, :], axis=0)
        node_coords -= mu
        node_coords[:, 1] *= shrinkage_correction

        if resolution is not None:
            X, region = rasterize(node_coords, connected_pairs, region, resolution)
        else:
            X = node_coords

        if correct_cut:
            # make sure the cut part is on the positive side
            if (X[:, 1] > 0).sum() > (X[:, 1] < 0).sum():
                X[:, 1] *= -1

            idx = X[:, 1] <= 0
            tmp = X[idx, :]
            tmp[:, 1] *= -1
            X = np.vstack((X, tmp))
            region = np.hstack((region, region[idx]))

        cells.append(X)
        regions.append(region)
        pairs.append(connected_pairs)
    if stack:
        return np.vstack(cells), np.hstack(regions), np.vstack(pairs), len(cells)
    else:
        return cells, regions, pairs, len(cells)


@schema
class CellRegion(dj.Lookup):
    definition = """
    # table for cell region labels

    cell_region_id      : tinyint   # cell region identifier
    ---

    cell_region_name    : char(8) # name of the cell region
    """

    contents = [
        (1, 'axon'), (2, 'cellbody'), (3, 'dendrite')
    ]


@schema
class DensityParameters(dj.Lookup):
    definition = """
    density_param_id        : tinyint            # identifier of the parameter set
    ---
    bin_width               : float              # bin width for the 3d histogram in mu
    raster_resolution       : float              # resolution of the rasterization in mu
    cell_offset             : float              # assumed offset between cells into the slice
    multiplication_factor   : int                # multiply the number of points by this factor when spin shuffling
    cut_compensation        : tinyint unsigned   # try to account for the fact that parts of the neuron might be missing
    shrinkage_correction    : float              # scale z-axis with that factor to account for shrinkage
    """

    contents = [
        dict(density_param_id=0, bin_width=15, raster_resolution=.5, cell_offset=0, multiplication_factor=1,
             cut_compensation=1, shrinkage_correction=300 / 160),
        dict(density_param_id=1, bin_width=15, raster_resolution=.5, cell_offset=0, multiplication_factor=10,
             cut_compensation=1, shrinkage_correction=300 / 160),
        dict(density_param_id=2, bin_width=15, raster_resolution=.5, cell_offset=50, multiplication_factor=10,
             cut_compensation=1, shrinkage_correction=300 / 160)
    ]


@schema
class CellNameTranslation(dj.Lookup):
    definition = """
    shan           : char(12)
    xiaolong       : char(12)
    paper          : char(12)
    xiaolong_new   : char(12)
    """
    contents = [('L1 eNGC', 'L1 eNGC', 'L1 eNGC', 'L1 eNGC'),
                ('L1 SBC', 'L1 non-eNGC', 'L1 SBC-like', 'L1 non-eNGC'),
                ('L23 BC', 'L23 BC', 'L23 BC', 'L23 BC'),
                ('L23 BPC', 'L23 BPC', 'L23 BPC', 'L23 BPC'),
                ('L23 BTC', 'L23 BTC', 'L23 BTC', 'L23 BTC',),
                ('L23 ChC', 'L23 ChC', 'L23 ChC', 'L23 ChC'),
                ('L23 DBC', 'L23 DBC', 'L23 DBC', 'L23 DBC'),
                ('L23 MaC', 'L23 MC', 'L23 MC', 'L23 MaC'),
                ('L23 NGC', 'L23 NGC', 'L23 NGC', 'L23 NGC'),
                ('L23 Pyr', 'L23 Pyr', 'L23 Pyr', 'L23 Pyr'),
                ('L5 BC', 'L5 BC', 'L5 BC', 'L5 BC'),
                ('L5 DC', 'L5 DC', 'L5 DC', 'L5 DC'),
                ('L5 HEC', 'L5 HEC', 'L5 HEC', 'L5 HEC'),
                ('L5 MaC', 'L5 MC', 'L5 MC', 'L5 MaC'),
                ('L5 NGC', 'L5 NGC', 'L5 NGC', 'L5 NGC'),
                ('L5 Pyr', 'L5 Pyr', 'L5 Pyr', 'L5 Pyr'),
                ('L5 SC', 'L5 Shrub', 'L5 SC', 'L5 SC')]


@schema
class PaperConnectivity(dj.Lookup):
    definition = """
    cell_type_from      : varchar(256)
    cell_type_to        : varchar(256)
    ---
    k       : int
    n       : int
    p       : float
    """

    @property
    def contents(self):
        labels, K, N = load_data()
        for (i, post), (j, pre) in product(enumerate(labels), enumerate(labels)):
            yield dict(cell_type_from=pre, cell_type_to=post, k=K[i, j], n=N[i, j], p=K[i, j] / N[i, j])


@schema
class CellTypePair(dj.Lookup):
    definition = """
    # table with cell type name pairs fetched from CellType

    cell_type_from     -> CellType[cell_type_name]
    cell_type_to       -> CellType[cell_type_name]
    ---
    """

    @property
    def contents(self):
        ct = CellType().fetch['cell_type_name']
        yield from product(ct, ct)


@schema
class OverlapGroup(dj.Computed):
    definition = """
    # Overlap densities grouping table
    -> CellTypePair
    -> DensityParameters
    ---
    """

    save_control_figures = True

    @property
    def key_source(self):
        return CellTypePair() * DensityParameters() & dict(density_param_id=1)

    class OrthoOverlapDensity(dj.Part):
        definition = """
        # overlap densities for a particular cell type pair and distance
        -> OverlapGroup
        -> conn.Distance
        ---
        d           : longblob # distance to the connecting line between two neurons
        p           : longblob # overlap density along that line
        """

    def _make_tuples(self, key):
        print('Populating', key)
        # get identifiers for different regions
        AXON = (CellRegion() & dict(cell_region_name='axon')).fetch1['cell_region_id']
        DENDRITE = (CellRegion() & dict(cell_region_name='dendrite')).fetch1['cell_region_id']

        # get parameters for density estimation
        raster_resolution, bin_width, cell_offset, multiplication_factor, cut_compensation, shrink_corr = \
            (DensityParameters() & key).fetch1['raster_resolution', 'bin_width',
                                               'cell_offset', 'multiplication_factor', \
                                               'cut_compensation', 'shrinkage_correction']

        # load all cells from the particular pair of cell types
        X, region, cells, N = {}, {}, {}, {}
        for role in ('from', 'to'):
            layer, morph = (CellType() & dict(cell_type_name=key['cell_type_' + role])).fetch1[
                'layer', 'cell_type_morph']

            cells[role] = (conn.ConnectMembership() * conn.Cell()
                           & dict(role=role, cell_layer=layer, cell_type_morph=morph)
                           ).proj(**{('cell_id_' + role): 'cell_id'})
            X[role], region[role], _, N[role] = load_cells(
                Tree() * CellType() & dict(layer=layer, cell_type_morph=morph), shrinkage_correction=shrink_corr,
                resolution=raster_resolution,
                correct_cut=cut_compensation)

        # if there is no cell pair recorded in the database, don't compute an overlap
        if not conn.Distance() * cells['from'] * cells['to']:
            self.insert1(key)
            return

        # get deltas
        keys, delta_x, delta_y = (conn.Distance() * cells['from'] * cells['to']).fetch[dj.key, 'delta_x', 'delta_y']
        offsets = np.c_[delta_x, -cell_offset * np.ones_like(delta_x), delta_y]

        if self.save_control_figures:
            if len(delta_x) > 1:
                sns.set(style="ticks")
                sns.jointplot(delta_x, delta_y, kind="scatter")
                plt.gca().set_xlabel(r'$\Delta$ lateral')
                plt.gca().set_ylabel(r'$\Delta$ depth')
                plt.gcf().suptitle('{cell_type_from} to {cell_type_to}'.format(**key))
                plt.gcf().tight_layout()
                plt.gcf().savefig(
                    'figures/sanity/distances/{cell_type_from}_to_{cell_type_to}_deltahist.png'.format(**key))

                plt.close(plt.gcf())

            fig, ax = plot_projections(X['from'], X['to'], offsets[0],
                                       dict(color=sns.xkcd_rgb['neon pink'], label=key['cell_type_from']),
                                       dict(color=sns.xkcd_rgb['neon blue'], label=key['cell_type_to']))
            fig.savefig('figures/sanity/cells/{cell_type_from}_to_{cell_type_to}.png'.format(**key))
            plt.close(fig)

        # shuffle by rotation around z-axis and increase data volume
        X['from'] = spin_shuffle(X['from'], copy=multiplication_factor)
        X['to'] = spin_shuffle(X['to'], copy=multiplication_factor)
        for role in region:
            region[role] = np.tile(region[role], multiplication_factor)

        # insert grouping key
        self.insert1(key)
        part = OverlapGroup.OrthoOverlapDensity()

        # insert an element in part (sub) table for each distance between cells of that type
        for k, delta in zip(keys, offsets):
            H, E = compute_overlap_density(X['from'][region['from'] == AXON], X['to'][region['to'] == DENDRITE],
                                           bin_width, delta, n=(N['from'], N['to']))

            # compute bin centers along y-axis and only get positive side
            y = 0.5 * (E[1][1:] + E[1][:-1])
            idx = y > 0
            # marginalize density
            p = H[:, idx, :].sum(axis=(0, 2))
            key.update({f: k[f] for f in k.dtype.fields})
            key.pop('cell_id_from')
            key.pop('cell_id_to')
            key['p'] = p
            key['d'] = y[idx]
            part.insert1(key)

    def plot_correction(self, filename, density_param_id=1, cut_distance=15):
        rel_correction = (self.OrthoOverlapDensity() & dict(density_param_id=density_param_id)) \
                         * CellNameTranslation().proj(cell_type_from='xiaolong_new', pre='paper', un1='shan', un2='xiaolong') \
                         * CellNameTranslation().proj(cell_type_to='xiaolong_new', post='paper',
                                         un3 = 'shan', un4 = 'xiaolong')
        rel_prob = PaperConnectivity() * CellNameTranslation().proj(cell_type_from='xiaolong', pre='paper',
                                                                    un1='shan', un2='xiaolong_new') \
                   * CellNameTranslation().proj(cell_type_to='xiaolong', post='paper', un3='shan', un4='xiaolong_new')

        df_correction = pd.DataFrame(rel_correction.fetch())
        df_paper = pd.DataFrame(rel_prob.fetch())
        P = df_paper.groupby(['post', 'pre']).agg({'p': lambda x: np.squeeze(x)}).copy()

        gr = df_correction.groupby(['post', 'pre'])

        def q(group):
            d = np.mean(group.d, axis=0)
            dd = np.diff(d)[0]
            lb = d - dd / 2
            idx = lb >= cut_distance
            return np.mean([x[idx].sum() / 2 / x.sum() if x.sum() > 0 else 0 for x in group.p])

        Q = (1 - gr.apply(q)).reset_index()
        Q.columns = ['post', 'pre', 'p']
        Q = Q.set_index(['post', 'pre'])
        # def q(X):
        #     return np.mean(
        #         [x[1:].sum() / 2 / x.sum() if x.sum() > 0 else 0 for x in X])  # TODO: replace 1: at some point

        # D0 = gr.agg({'p': lambda x: np.mean(x, axis=0).sum() * 2})
        # D = gr.agg({'p': lambda x: np.mean(x, axis=0)[1:].sum()})
        # Q = 1 - D / D0

        # Q = 1 - gr.agg({'p': q})
        print('<b> +- std(b)', (1 / Q).mean(), (1 / Q).std())
        labels = ['L1 SBC-like', 'L1 eNGC', 'L23 MC', 'L23 NGC', 'L23 BTC', 'L23 BPC', 'L23 DBC', 'L23 BC', 'L23 ChC',
                  'L23 Pyr', 'L5 MC', 'L5 NGC', 'L5 BC', 'L5 SC', 'L5 HEC', 'L5 DC', 'L5 Pyr']
        P2 = (P / Q).fillna(0)
        P = P.unstack().loc[labels][list(product(['p'], labels))]
        P2 = P2.unstack().loc[labels][list(product(['p'], labels))]

        fig, axes = plot_connections(P, P2, vmax=1, cmin=0, cmax=1)
        axes['matrix'][0].set_title('uncorrected', fontsize=6)
        axes['matrix'][1].set_title('corrected', fontsize=6)
        fig.tight_layout()

        p2 = df_paper.merge((1 / Q).reset_index(), on=['post', 'pre'], suffixes=('_prop', '_correction'))
        axes['correlation'].plot(p2.p_prop, p2.p_correction, 'ow', ms=2, color='slategray', lw=1)
        print(np.corrcoef(p2.p_prop, p2.p_correction))
        axes['correlation'].set_xlim((-0.05, 0.7))
        axes['correlation'].set_ylim((0.95, 1.75))

        axes['correlation'].set_yticks([1., 1.3, 1.6])
        axes['correlation'].set_xticks(np.arange(0, 1, .2))
        for _, i in axes['correlation'].spines.items():
            i.set_linewidth(0.5)
        axes['correlation'].tick_params(axis='both', which='major', labelsize=6, length=3, width=.5)
        axes['correlation'].set_xlabel('connection probability', fontsize=6)
        axes['correlation'].set_ylabel('correction', fontsize=6)
        sns.despine(ax=axes['correlation'], offset=3, trim=True)
        fig.savefig(filename)

    def overlap_sanity(self, filename, density_param_id=1):
        rel_correction = (self.OrthoOverlapDensity() * conn.Distance() & dict(density_param_id=density_param_id)) \
                         * CellNameTranslation().proj(cell_type_from='shan', pre='paper', un1='xiaolong') \
                         * CellNameTranslation().proj(cell_type_to='shan', post='paper', un2='xiaolong')
        rel_prob = PaperConnectivity() * CellNameTranslation().proj(cell_type_from='xiaolong', pre='paper',
                                                                    un3='shan') \
                   * CellNameTranslation().proj(cell_type_to='xiaolong', post='paper', un4='shan')

        df_correction = pd.DataFrame(rel_correction.fetch())
        df_paper = pd.DataFrame(rel_prob.fetch())
        df = df_correction.merge(df_paper, on=['post', 'pre'])
        df2 = df.ix[:, ('p_y', 'distance', 'q', 'pre', 'post')]
        df2['q'] = [x[1:].sum() / x.sum() / 2 if x.sum() > 0 else 0 for x in df.p_x]
        for field in ('p_y', 'distance', 'q'):
            df2[field] = np.asarray(df2[field]).astype(np.float64)
        gr = df2.groupby(['post', 'pre'])
        mu = gr.mean()
        s = gr.std()

        with sns.axes_style('whitegrid'):
            fig, ax = plt.subplots()
        cb = ax.scatter(mu.distance, s.q, c=mu.p_y, cmap=plt.cm.get_cmap('viridis'), s=50)
        plt.colorbar(cb)
        ax.set_ylim((0, .3))
        ax.set_ylabel('std of correction factor')
        ax.set_xlabel('mean distance between cell types')
        ax.set_title("Color is Xiaolong's connection probability")

        fig.savefig(filename)

    def plot_schematic(self, filename_base, fro='L23 MaC', to='L23 BC'):
        cm = plt.cm.get_cmap('bwr')
        cm._init()
        cm._lut[:-3, -1] = np.abs(np.linspace(0, 1.0, cm.N))

        axon_color = sns.xkcd_rgb['azure']
        dendrite_color = sns.xkcd_rgb['purple pink']
        AXON = (CellRegion() & dict(cell_region_name='axon')).fetch1['cell_region_id']
        DENDRITE = (CellRegion() & dict(cell_region_name='dendrite')).fetch1['cell_region_id']

        # load all cells from the particular pair of cell types
        X, region, cells, skeleton = {}, {}, {}, {}
        for role, tp in zip(('from', 'to'), (fro, to)):
            layer, morph = (CellType() & dict(cell_type_name=tp)).fetch1['layer', 'cell_type_morph']

            cells[role] = (conn.ConnectMembership() * conn.Cell()
                           & dict(role=role, cell_layer=layer, cell_type_morph=morph)
                           ).proj(**{('cell_id_' + role): 'cell_id'})
            X[role], region[role], skeleton[role], _ = load_cells(Tree() * CellType()
                                                                  & dict(layer=layer, cell_type_morph=morph),
                                                                  resolution=None, correct_cut=False,
                                                                  stack=False, shrinkage_correction=300 / 160)
            for i in range(len(X[role])):  # flip y axis for easier plotting
                X[role][i][:, 1] *= -1

        keys, delta_x, delta_y = (conn.Distance() * cells['from'] * cells['to']).fetch[dj.key, 'delta_x', 'delta_y']
        offsets = np.c_[delta_x, np.zeros_like(delta_x), delta_y]
        delta = 3 * offsets[0]

        fig = plt.figure(facecolor='w', dpi=400)
        ax = fig.add_subplot(1, 1, 1, projection='3d', axisbg='w')
        x_fro, x_to = X['from'][0], X['to'][0]
        plot_skeleton(ax, x_fro, skeleton['from'][0], region['from'][0] == AXON,
                      mask_kw=dict(lw=.2, ms=2, color=axon_color, label=fro + ' axon'),
                      other_kw=dict(lw=.1, ms=1, color='grey', label=fro + ' dendrite'), fast=False, stride=1
                      )
        plot_skeleton(ax, x_to + delta, skeleton['to'][0], region['to'][0] == DENDRITE,
                      mask_kw=dict(lw=.2, ms=2, color=dendrite_color, label=to + ' dendrite'),
                      other_kw=dict(lw=.1, ms=1, color='grey', label=to + ' axon'), fast=False, stride=1
                      )
        # make rotationally symmetric
        x_fro = spin_shuffle(x_fro, 10)
        x_to = spin_shuffle(x_to, 10) + delta

        # compute 3d histograms and overlap density
        tmp = np.vstack((x_fro, x_to))
        x_min, y_min, z_min = tmp.min(axis=0)
        x_max, y_max, z_max = tmp.max(axis=0)
        db = 10
        bins = (np.arange(x_min, x_max + db, db),
                np.arange(y_min, y_max + db, db),
                np.arange(z_min, z_max + db, db))

        H_from, _ = np.histogramdd(x_fro, bins=bins)
        H_to, __ = np.histogramdd(x_to, bins=bins)
        H_to /= db ** 3
        H_from /= db ** 3

        # compute marginal density and scale for plot
        f = (H_from * H_to).sum(axis=(0, 2))
        t = bins[1]
        t = 0.5 * (t[1:] + t[:-1])
        f = f / f.max() * 200
        f += z_max
        idx = (t >= -125) & (t <= 300 - 15)
        ax.plot(0 * t[idx], t[idx], f[idx], color='k', lw=1)
        t1 = np.linspace(-15, 300 - 15, 100)
        f1 = interp(t1, t, f)
        idx = (t >= -15) & (t <= 300 - 15)
        t, f = t1, f1  # t[idx], f[idx]
        # plot marginal density as filled curve in 3d
        v = []

        for k in range(0, len(t) - 1):
            v.append(list(zip(
                np.zeros(4),
                [t[k], t[k + 1], t[k + 1], t[k]],
                [f[k], f[k + 1], z_max, z_max])))

        poly3dCollection = Poly3DCollection(v, linewidths=0, facecolors=sns.xkcd_rgb['golden rod'], alpha=.5)
        ax.add_collection3d(poly3dCollection)

        # plot single log-densities and log product density
        H_from = H_from.sum(axis=2)
        H_to = H_to.sum(axis=2)

        H = np.log(1e-2 + H_from * H_to)

        x, y = np.meshgrid(*map(lambda x: 0.5 * (x[1:] + x[:-1]), bins[:2]))
        vs = [-4.75, -4.25, -3.75, -3.25, -2.75, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25]
        H[H < vs[1]] = np.nan
        ax.contourf(x.T, y.T, H, vs, zdir='z', offset=z_max, cmap=cm)

        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        ax.contour(x.T, y.T, np.log(1e-1 + H_from), 5, zdir='z', offset=z_min, colors=axon_color, linewidths=.8,
                   alpha=.5)
        ax.contour(x.T, y.T, np.log(1e-1 + H_to), 5, zdir='z', offset=z_min + 1, colors=dendrite_color,
                   linewidths=.8, alpha=.5)

        # plot slice boundaries
        x, z = np.meshgrid(np.linspace(x_min, x_max, 3), np.linspace(z_min, z_max, 3))
        ax.plot_surface(x, 0 * x - 15, z, color='silver', alpha=.1)
        ax.plot_surface(x, 0 * x - 15 + 300, z, color='silver', alpha=.1)
        y, z = np.meshgrid(np.linspace(-15, 300 - 15, 3), np.linspace(z_min, z_max, 3))
        ax.plot_surface(0 * y + x_max, y, z, color='silver', alpha=.1)
        ax.plot_surface(0 * y + x_min, y, z, color='silver', alpha=.1)

        # plot cosmetics
        ax.set_zlim((z_min, z_max))
        ax.set_aspect(1)
        ax.view_init(elev=29, azim=-43)
        ax.axis('equal')
        ax.axis('off')

        for form in ['png', 'pdf']:
            fig.savefig('{3}_from_{0}_to_{1}.{2}'.format(fro, to, form, filename_base))
