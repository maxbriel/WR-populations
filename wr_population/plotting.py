"""Shared plotting helpers and colour conventions."""
from pathlib import Path

import numpy as np


def use_posydon_style():
    """Apply the POSYDON matplotlib style (LaTeX fonts) used for the paper figures."""
    import matplotlib.pyplot as plt
    import posydon
    plt.style.use(str(Path(posydon.__file__).parent / 'visualization' / 'posydon.mplstyle'))

# colours per metallicity (in units of Zsun)
METALLICITY_COLOURS = {0.0001: '#1f4e79',
                       0.001: '#2e75b6',
                       0.01: '#5ba3d0',
                       0.1: '#f4a261',
                       0.2: '#e76f51',
                       0.45: '#c0392b',
                       1.0: '#8e1a0e',
                       2.0: '#5c0a0a'}


def plot_period_histogram(ax, periods, n_bins=50, log_p_range=(-1, 6), scale=1.0,
                          color='black', fill_alpha=0.2, linewidth=1.0, label=None, **kwargs):
    """Plot an observed period distribution as a density histogram in log10(P).

    The density is multiplied by `scale` (e.g. to compare with a model PDF).
    """
    bins = np.linspace(log_p_range[0], log_p_range[1], n_bins + 1)
    h, edges = np.histogram(np.log10(np.asarray(periods, dtype=float)), bins=bins, density=True)
    ax.fill_between(10**edges[:-1], h * scale, step='post', alpha=fill_alpha, color=color,
                    zorder=10, label=label, edgecolor='none', **kwargs)
    ax.step(10**edges[:-1], h * scale, where='post', color=color, linewidth=linewidth,
            zorder=11, alpha=0.6)
