"""WR sub-type classification and formation-channel masks."""
import numpy as np
import pandas as pd
from cmap import Colormap

from .io import load_interaction_classes

BINARY_STATES = ['detached', 'RLO1', 'RLO2', 'contact']


def classify_WR(d, tau_min=1.5, exclude_H_burning=True):
    """Return WNL, WNE, WC boolean masks for a data dict from `load_metallicity`.

    WNL: 0.05 < X_H <= 0.5 (and not core-H burning, unless exclude_H_burning=False)
    WNE: X_H <= 0.05, X_He >= 0.7
    WC:  X_H <= 0.05, X_He < 0.7
    All require an optical depth tau > tau_min.
    """
    thick = d['optical_depth'] > tau_min
    WNL = (d['surface_H'] > 0.05) & (d['surface_H'] <= 0.5) & thick
    if exclude_H_burning:
        WNL &= ~pd.Series(d['star_state']).str.contains('H_burning', na=False).values
    WNE = (d['surface_H'] <= 0.05) & (d['surface_He'] >= 0.7) & thick
    WC = (d['surface_H'] <= 0.05) & (d['surface_He'] < 0.7) & thick
    return WNL, WNE, WC


def make_binary_mask(d):
    """Return mask for detached/RLO/contact binaries."""
    return np.isin(d['state'], BINARY_STATES)


# Formation channels: S1 is classified by its HMS-HMS mass transfer, S2 by its
# CO-HMS mass transfer. A missing class (NaN) means the binary never went
# through that step, which we count as no mass transfer.
S1_CHANNELS = {'stable_MT': 'Stable (S1)',
               'unstable_MT': 'Unstable (S1)',
               'stable_reverse_MT': 'Reverse Stable (S1)',
               'no_MT': 'No MT (S1)'}
S2_CHANNELS = {'stable_MT': 'Stable (S2)',
               'unstable_MT': 'Unstable (S2)',
               'no_MT': 'No MT (S2)'}

CHANNEL_LABELS = ['Stable (S1)', 'Stable (S2)', 'Unstable (S1)', 'Unstable (S2)',
                  'No MT (S1)', 'No MT (S2)', 'Reverse Stable (S1)']

_paired = Colormap('colorbrewer:Paired_8')(np.linspace(0, 1, 8))
CHANNEL_COLOURS = {'Stable (S1)': _paired[1],
                   'Stable (S2)': _paired[0],
                   'Unstable (S1)': _paired[3],
                   'Unstable (S2)': _paired[2],
                   'No MT (S1)': _paired[5],
                   'No MT (S2)': _paired[4],
                   'Reverse Stable (S1)': _paired[7]}


def formation_channels(d, metallicity, interaction_classes=None):
    """Formation-channel label of every row of a data dict from `load_metallicity`.

    Rows whose interaction class is not in S1_CHANNELS/S2_CHANNELS get 'Other'.
    """
    if interaction_classes is None:
        interaction_classes = load_interaction_classes(metallicity)
    channel = np.full(len(d['key']), 'Other', dtype=object)
    for star, column, channels in [('S1', 'interp_class_HMS_HMS', S1_CHANNELS),
                                   ('S2', 'interp_class_CO_HMS_RLO', S2_CHANNELS)]:
        rows = d['star'] == star
        interaction = (interaction_classes[column]
                       .reindex(d['key'][rows]).fillna('no_MT').values)
        channel[rows] = pd.Series(interaction).map(channels).fillna('Other').values
    return channel


def formation_channel_masks(d, metallicity, interaction_classes=None):
    """Dict of formation-channel label -> boolean mask over the rows of `d`."""
    channel = formation_channels(d, metallicity, interaction_classes)
    return {label: channel == label for label in CHANNEL_LABELS}
