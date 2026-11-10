"""Shared, reproducible visual style for the Visuelle analysis notebooks."""
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.container import BarContainer

INK = '#20334B'
MUTED = '#63748B'
TEAL = '#168F91'
CORAL = '#E47759'
NAVY = '#345A87'
GOLD = '#C89B36'
PAL = ['#219EBC', '#F28E2B', '#8A60B0', '#2E75B6', '#E5B83D', '#E36B83', '#38A67E', '#D65A4A']
SEASON_COLORS = PAL[:6]
COLOR_MAP = {'black': '#344052', 'brown': '#A27D61', 'white': '#E8EDF3',
             'blue': '#5686BA', 'grey': '#9DA9B6', 'violet': '#9774B0',
             'green': '#67A18B', 'red': '#D66E72', 'yellow': '#DFBC55', 'orange': CORAL}
SEQ = LinearSegmentedColormap.from_list('visuelle_sunset',
    ['#FFF1AD', '#FFC16E', '#F58465', '#CE527E', '#8754A2', '#414E92', '#23385D'])


def categorical_colors(n):
    return [PAL[i % len(PAL)] for i in range(n)]


def apply_theme(journal=False):
    size = 7.5 if journal else 10
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.size': size, 'axes.labelsize': size, 'axes.titlesize': size + 1,
        'axes.titleweight': 'bold', 'axes.titlepad': 12, 'axes.titlelocation': 'left',
        'xtick.labelsize': size - .6, 'ytick.labelsize': size - .6,
        'legend.fontsize': size - .7, 'legend.frameon': False,
        'legend.handlelength': 1.8, 'legend.columnspacing': 1.1,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#B7C4D1',
        'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.facecolor': '#FFFFFF', 'figure.facecolor': '#FFFFFF',
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'axes.axisbelow': True, 'axes.grid': False,
        'grid.color': '#E5ECF2', 'grid.linewidth': .55,
        'axes.prop_cycle': mpl.cycler(color=PAL),
        'lines.linewidth': 1.6 if journal else 2.3,
        'lines.markersize': 3.8 if journal else 5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.dpi': 150 if journal else 110, 'savefig.dpi': 600,
        'savefig.bbox': 'tight', 'savefig.pad_inches': .12,
        'figure.constrained_layout.w_pad': .06,
        'figure.constrained_layout.h_pad': .09,
        'figure.constrained_layout.wspace': .08,
        'figure.constrained_layout.hspace': .12,
    })


def ramp(n, color=TEAL):
    """A muted tonal progression, preserving a single series' meaning."""
    base = np.array(to_rgb(color))
    return [tuple(base * t + np.ones(3) * (1-t)) for t in np.linspace(.48, 1, n)]


def hgrid(ax, axis='y'):
    ax.grid(axis=axis, color='#E5ECF2', linewidth=.55)
    ax.set_axisbelow(True)


def panel_title(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', fontsize=8.2,
                 fontweight='bold', color=INK, pad=11)


def finish_eda(fig):
    """Polish notebook charts and distinguish groups without changing their data."""
    for ax in fig.axes:
        if ax.images or not ax.axison:
            continue
        bars = [c for c in ax.containers if isinstance(c, BarContainer)]
        for container in bars:
            if not container.patches:
                continue
            colors = {tuple(p.get_facecolor()) for p in container.patches}
            if len(colors) == 1:
                # Distinct hues make separate bins / groups visually distinguishable.
                count = len(container.patches)
                shades = [SEQ(v) for v in np.linspace(.05, .9, count)] if count > 30 else categorical_colors(count)
                for p, color in zip(container.patches, shades):
                    p.set_facecolor(color)
                    p.set_edgecolor('white')
                    p.set_linewidth(.65)
        ax.spines['left'].set_color('#B7C4D1')
        ax.spines['bottom'].set_color('#B7C4D1')
        horizontal = any(getattr(c, 'orientation', None) == 'horizontal' for c in bars)
        hgrid(ax, 'x' if horizontal else 'y')
        ax.set_title(ax.get_title(loc='left'), loc='left', pad=14)
    # Colorbars already created with the default layout have their own nested
    # gridspec. Switching those figures to constrained layout can divide by zero.
    if any(ax.get_label() == '<colorbar>' for ax in fig.axes):
        fig.tight_layout(pad=1.5)
    else:
        fig.set_layout_engine('constrained')
