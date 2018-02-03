"""
Helper function for plotting. All function plot into a given axis
which makes it easy to combine different plots into a single
figure. Exceptions make the rule.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from rootpy.plotting import Hist2D, Canvas, Pad
from root_numpy import array2hist

from ROOT import TLatex, gStyle

from c2_post import utils


# Latex Book: \textwidth = 390pt
# 1pt = 72.72 pt/inch
# in my thesis: 4.9823
TEXT_WIDTH = 4.9823


def load_matplotlib_defaults():
    mpl.rc('font', family='sans-serif', serif='helvet')
    mpl.rc('text', usetex=True)
    # This ensures that no lables are cut of, but the size of the figure changes
    mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams['text.latex.preamble'] = [
        # It seems that siunitx keeps using serif fonts for numbers!
        # r'\usepackage{amsmath}',
        r'\usepackage{helvet}',    # set the normal font here
        r'\usepackage{tgheros}',   # upper case greek letters
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
    ]
    params = {'legend.fontsize': 'small',
              'axes.labelsize': 'small',
              'axes.titlesize': 'small',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'small'}
    mpl.rcParams.update(params)


load_matplotlib_defaults()


def restore_matplotlib_defaults():
    mpl.rcParams.update(mpl.rcParamsDefault)


def generic_2d_no_cbar(a, edges1, edges2, ax, label=r'', a_label=r'', b_label=r'', **kwargs):
    """
    Plot a 2D heatmap without a color bar
    """
    # Mask invalid values
    a = np.ma.masked_invalid(a)
    mappable = ax.pcolormesh(edges2, edges1, a,
                             cmap=plt.get_cmap(kwargs.get('cmap', 'viridis')),
                             vmin=kwargs.get('vmin', None),
                             vmax=kwargs.get('vmax', None))
    # Legend for invalid data if any
    has_legend = False
    if np.any(np.isnan(a)) or (np.ma.is_masked(a) and np.any(~a.mask)):
        nan_color = 'gray'
        ax.patch.set(facecolor=nan_color)
        patch_label = mpl.patches.Patch(color=nan_color, label='No data')
        ax.legend(handles=[patch_label], loc='upper left')
        has_legend = True
    # its row - colum order in numpy!
    ax.set_xlabel(b_label)
    ax.set_ylabel(a_label)
    ax.minorticks_on()
    if kwargs.get('title', None):
        plt.title(kwargs.get('title', None))
    if has_legend:
        # Otherwise, we get a warning and an empty legend box
        # ax.legend()
        pass
    return mappable


def generic_2d(a, edges1, edges2, ax=None, label=r'', a_label=r'', b_label=r'', cax=None, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=(TEXT_WIDTH / 2 * 1.25, TEXT_WIDTH / 2))
        gs = mpl.gridspec.GridSpec(2, 15, wspace=0.2, top=0.95, bottom=0.15, left=0.15, right=0.88)
        fig.tight_layout()
        ax = plt.subplot(gs[:, :-1])
        cax = plt.subplot(gs[:, -1:])
    mappable = generic_2d_no_cbar(a, edges1, edges2, ax, label, a_label, b_label, **kwargs)
    if cax is None:
        # allocates the colorbar axis; (extending the figure?)
        # numbers are magic to make the cbar have the same hight as the plot
        plt.colorbar(mappable, fraction=0.046, pad=0.03, label=label)
    else:
        plt.colorbar(mappable, label=label, cax=cax)
    return ax


def prettify_phi_axis(axis):
    """
    Make the ticks of the given axis be in multiples of pi/2
    """
    axis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    axis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    axis.set_major_formatter(plt.FuncFormatter(format_func))


def plot_dphi(a, centers, ax=None, sigma=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if sigma is not None:
        yerr = sigma / 2
    else:
        yerr = 0
    ax.errorbar(
        x=centers,
        y=a,
        yerr=yerr,
        marker=kwargs.get('marker', 'o')
    )
    ax.set_xlabel(r"$\Delta\varphi$")
    ax.set_ylabel(r"$c_2(\Delta\varphi)$")
    return ax


def plot_cms(alice_icent, hcal_iseg, ax=None):
    if not ax:
        ax = plt.gca()
    xval = [0.15, 0.45, 0.75, 1.05, 1.35, 1.65, 1.95, 2.25]

    if hcal_iseg == 0:
        label = r'CMS: $3.0<\eta_b<4.0$'
    elif hcal_iseg == 1:
        label = r'CMS: $4.4<\eta_b<5.0$'
    else:
        raise NotImplementedError("Unrecognized hcal segment {}".format(hcal_iseg))

    if alice_icent == 0 and hcal_iseg == 0:
        yval = [0.9957, 1.0012, 0.9924, 0.9767, 0.9648, 0.9348, 0.91, 0.8648]
        yerr = [0.0035, 0.0035, 0.0037, 0.0041, 0.0051, 0.0051, 0.0062, 0.0096]

    elif alice_icent == 1 and hcal_iseg == 0:
        yval = [0.9986, 0.9946, 0.9898, 0.9835, 0.9777, 0.9653, 0.9477, 0.9287]
        yerr = [0.0015, 0.0015, 0.0016, 0.0017, 0.0021, 0.0021, 0.0025, 0.0038]

    elif alice_icent == 2 and hcal_iseg == 0:
        # Centrality 50-60%
        print "ALICE and CMS cent bins do not match for this centrality bin!"
        yval = [0.9978, 0.9894, 0.9817, 0.9747, 0.9606, 0.9484, 0.9341, 0.9156]
        yerr = [0.0017, 0.0017, 0.0017, 0.0018, 0.002, 0.0022, 0.0024, 0.0032]

    elif alice_icent == 0 and hcal_iseg == 1:
        yval = [0.9972, 0.9946, 0.9905, 0.9856, 0.9788, 0.9679, 0.9676, 0.9499]
        yerr = [0.0035, 0.0035, 0.0037, 0.004, 0.0049, 0.0049, 0.0063, 0.0101]

    elif alice_icent == 1 and hcal_iseg == 1:
        yval = [0.9973, 0.9943, 0.9919, 0.9847, 0.9826, 0.9772, 0.9705, 0.9647]
        yerr = [0.0018, 0.0018, 0.0018, 0.002, 0.0023, 0.0024, 0.0029, 0.0045]

    elif alice_icent == 2 and hcal_iseg == 1:
        # Centrality 50-60%
        print "ALICE and CMS cent bins do not match for this centrality bin!"
        yval = [0.995, 0.9885, 0.9836, 0.9742, 0.9672, 0.9583, 0.9528, 0.9429]
        yerr = [0.0021, 0.0021, 0.0021, 0.0022, 0.0024, 0.0026, 0.0029, 0.0038]

    else:
        raise NotImplementedError("Unrecognized icent bin {}".format(alice_icent))

    ax.errorbar(xval, yval, yerr=yerr, label=label, ls='None', marker='s')


def format_func(value, tick_number):
    """Find number of multiples of pi/2

    See: https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
    """
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi / 2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)
