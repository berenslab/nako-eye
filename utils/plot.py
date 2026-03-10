# From https://github.com/jonathanoesterle/data_plots/tree/master
import io
import os

import cairosvg
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image

# Size of figures
_FIG_WIDTHS = dict(
    col=3.14,  # Single column in bioRxiv template
    full=6.27,  # Full width in word
    slide_col=5.67,  # Half powerpoint slide
    slide_full=11.5,  # Full powerpoint slide
    poster=14.46,  # Full poster width
)


def set_rc_params(kind, notebook_dpi=None):
    # Arial ttf file sould be in the home dir
    font_dir = os.path.expanduser('~/.fonts/mscorefonts')
    for fname in os.listdir(font_dir):
        if fname.lower().endswith('.ttf'):
            fm.fontManager.addfont(os.path.join(font_dir, fname))

    sns.set_context('paper')
    sns.set_style('ticks')
    plt.style.use(os.path.join(os.getcwd(), 'mplstyles', f'{kind}.mplstyle'))
    if notebook_dpi is not None:
        plt.rcParams['figure.dpi'] = notebook_dpi


def set_figsize(fig, width, height_ratio=None):
    if isinstance(width, str):
        if width not in _FIG_WIDTHS:
            raise NotImplementedError(f'Unknown width `{width}`')
        else:
            width = _FIG_WIDTHS[width]

    fig.set_figwidth(width)

    if height_ratio is not None:
        fig.set_figheight(width * height_ratio)


def load_svg(image_file, return_np=True):
    png_bytes = cairosvg.svg2png(url=image_file, dpi=300)
    image = Image.open(io.BytesIO(png_bytes))
    image.load()
    data = np.asarray(image, dtype='int32')
    return data if return_np else image


def iterate_axes(axs):
    """Make axes iterable, independent of type.
    axs (list of matplotlib axes or matplotlib axis) : Axes to apply function to.
    """

    if isinstance(axs, list):
        return axs
    elif isinstance(axs, np.ndarray):
        return axs.flatten()
    else:
        return [axs]


def grid(ax, axis='both', major=True, minor=False, **kwargs):
    """Make grid on axis"""
    for axi in iterate_axes(ax):
        if major:
            axi.grid(
                True,
                axis=axis,
                which='major',
                alpha=0.3,
                c='k',
                lw=plt.rcParams['ytick.major.width'],
                zorder=-10000,
                **kwargs,
            )
        if minor:
            axi.grid(
                True,
                axis=axis,
                which='minor',
                alpha=0.3,
                c='gray',
                lw=plt.rcParams['ytick.minor.width'],
                zorder=-20000,
                **kwargs,
            )


def dettach_axes(ax, coord=3):
    ax.spines['left'].set_position(('outward', coord))
    ax.spines['bottom'].set_position(('outward', coord))
    ax.tick_params(direction='out')


def tight_layout(h_pad=1, w_pad=1, rect=(0, 0, 1, 1), pad=None):
    """Like `tight_layout` with different default"""
    plt.tight_layout(
        h_pad=h_pad, w_pad=w_pad, pad=pad or 2.0 / plt.rcParams['font.size'], rect=rect
    )


def set_labs(
    axs,
    xlabs=None,
    ylabs=None,
    titles=None,
    panel_nums=None,
    panel_num_space=0,
    panel_num_va='bottom',
    panel_num_pad=0,
    panel_num_y=None,
):
    """Set labels and titles for all given axes.
    Parameters:

    axs : array or list of matplotlib axes.
        Axes to apply function to.

    xlabs, ylabs, titles : str, list of str, or None
        Labels/Titles.
        If single str, will be same for all axes.
        Otherwise, should have same length as axes.

    """

    for i, ax in enumerate(iterate_axes(axs)):
        if xlabs is not None:
            if isinstance(xlabs, str):
                xlab = xlabs
            else:
                xlab = xlabs[i]
            ax.set_xlabel(xlab)

        if ylabs is not None:
            if isinstance(ylabs, str):
                ylab = ylabs
            else:
                ylab = ylabs[i]
            ax.set_ylabel(ylab)

        if titles is not None:
            if isinstance(titles, str):
                title = titles
            else:
                title = titles[i]
            ax.set_title(title)

        if panel_nums is not None:
            if panel_nums == 'auto':
                panel_num = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]
            elif isinstance(panel_nums, str):
                panel_num = panel_nums
            else:
                panel_num = panel_nums[i]
            ax.set_title(
                panel_num + panel_num_space * ' ',
                loc='left',
                fontweight='bold',
                ha='right',
                va=panel_num_va,
                pad=panel_num_pad,
                y=panel_num_y,
            )
