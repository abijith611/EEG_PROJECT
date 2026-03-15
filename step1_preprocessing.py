# step1_preprocessing.py
"""
Pre-processing script:
  - Load raw .bdf data and events
  - Epoch data (-0.2 to 5.0s)
  - Custom distance-based neighbor interpolation (inverse‑distance weighting, 0.05 m)
  - Down-sample to 256 Hz
  - Save as -epo.fif with standard 10‑20 channel names
"""

import os
import mne
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Optional
from config import PATH_TO_DATA, DERIV_DIR, PAIR_IDS, MATLAB_LAYOUT_LABELS, SFREQ, get_logger

logger = get_logger(__name__)


def repair_bads_inverse_distance(epochs: mne.Epochs, bad_chans: List[str], thresh: float = 0.05) -> mne.Epochs:
    """
    Replicate MATLAB's ft_channelrepair with method='distance'.

    For each bad channel, find neighbours within `thresh` meters,
    compute inverse‑distance weights, and replace the bad channel's data
    with the weighted average of its neighbours.
    If no neighbours are found within the threshold, fall back to using all
    good channels (with inverse‑distance weighting based on all distances).

    Args:
        epochs: Epochs object with bad channels marked in `bad_chans`.
        bad_chans: Names of channels to repair (must match epochs.ch_names).
        thresh: Distance threshold in meters (default 0.05 = 5 cm).

    Returns:
        epochs: The same epochs object with data repaired (modified in‑place).
    """
    # Get channel positions
    pos = epochs.get_montage().get_positions()['ch_pos']
    ch_names = epochs.ch_names
    pos_arr = np.array([pos[name] for name in ch_names])
    dist_mat = cdist(pos_arr, pos_arr)  # meters

    # Identify good channels (not in bad list)
    good_mask = np.array([ch not in bad_chans for ch in ch_names])
    good_idx = np.where(good_mask)[0]
    if len(good_idx) == 0:
        raise RuntimeError("No good channels left to interpolate from.")

    data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)

    for bad in bad_chans:
        if bad not in ch_names:
            logger.warning(f"Channel {bad} not found, skipping.")
            continue
        bad_idx = ch_names.index(bad)

        # Find neighbours within threshold (excluding itself)
        within_thresh = np.where((dist_mat[bad_idx] < thresh) &
                                 (dist_mat[bad_idx] > 0))[0]
        # Restrict to good channels only
        neigh_idx = np.intersect1d(within_thresh, good_idx)

        if len(neigh_idx) == 0:
            # Fallback: use all good channels (distance‑based weights)
            neigh_idx = good_idx
            dists = dist_mat[bad_idx, neigh_idx]
        else:
            dists = dist_mat[bad_idx, neigh_idx]

        logger.info(f"repairing channel {bad}")
        for nb_idx in neigh_idx:
            logger.info(f"    using neighbour {ch_names[nb_idx]}")

        # Compute inverse‑distance weights
        with np.errstate(divide='ignore'):
            weights = 1.0 / dists
        weights[np.isinf(weights)] = 0
        weights /= weights.sum()

        # Weighted average across neighbours
        data[:, bad_idx, :] = np.average(data[:, neigh_idx, :], axis=1, weights=weights)

    logger.info(f"interpolating bad channels for {len(epochs)} trials.")
    epochs._data = data
    return epochs


def run_preprocessing(max_pairs: Optional[int] = None) -> None:
    """
    Run preprocessing for all specified pairs.

    Args:
        max_pairs: If given, process only the first `max_pairs` pairs (for testing).
    """
    os.makedirs(DERIV_DIR, exist_ok=True)

    participants = pd.read_csv(os.path.join(PATH_TO_DATA, 'participants.tsv'), sep='\t')
    pairs_to_run = PAIR_IDS[:max_pairs] if max_pairs is not None else PAIR_IDS
    logger.info(f"Processing {len(pairs_to_run)} pairs: {pairs_to_run}")

    for pair in pairs_to_run:
        sub_str = f'sub-{pair:02d}'
        eeg_dir = os.path.join(PATH_TO_DATA, sub_str, 'eeg')

        raw_path = os.path.join(eeg_dir, f'{sub_str}_task-RPS_eeg.bdf')
        if not os.path.exists(raw_path):
            logger.error(f"Raw file not found: {raw_path}")
            continue
        raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose=False)

        events_path = os.path.join(eeg_dir, f'{sub_str}_task-RPS_events.tsv')
        if not os.path.exists(events_path):
            logger.error(f"Events file not found: {events_path}")
            continue
        events_df = pd.read_csv(events_path, sep='\t')

        montage = mne.channels.make_standard_montage('biosemi64')

        for ppt in [1, 2]:
            out_file = os.path.join(DERIV_DIR, f'pair-{pair:02d}_player-{ppt}_task-RPS-epo.fif')
            if os.path.exists(out_file):
                logger.info(f"Skipping pair {pair} player {ppt} – output already exists.")
                continue

            prefix = '2-' if ppt == 1 else '1-'
            ppt_chans = [ch for ch in raw.ch_names if (ch.startswith(prefix + 'A') or ch.startswith(prefix + 'B'))]
            if not ppt_chans:
                logger.error(f"No channels found for player {ppt} in pair {pair}. Skipping.")
                continue
            raw_ppt = raw.copy().pick(ppt_chans)

            # Force the MATLAB channel scrambling: map current hardware channels sequentially
            current_chans = raw_ppt.ch_names
            forced_mapping = {current_chans[i]: MATLAB_LAYOUT_LABELS[i] for i in range(len(current_chans))}
            raw_ppt.rename_channels(forced_mapping)
            raw_ppt.set_montage(montage, on_missing='ignore')

            # Create events array (timing only)
            mne_events = np.column_stack((
                events_df['onset_sample'].values,
                np.zeros(len(events_df), dtype=int),
                np.ones(len(events_df), dtype=int)
            ))

            # Epoch relative to decision onset
            epochs = mne.Epochs(raw_ppt, mne_events, tmin=-0.2, tmax=5.0,
                                baseline=None, preload=True, verbose=False)

            # Sanity check: number of epochs
            if len(epochs) == 0:
                logger.error(f"No epochs created for pair {pair} player {ppt}. Skipping.")
                continue
            logger.info(f"Created {len(epochs)} epochs for pair {pair} player {ppt}.")

            # Repair bad channels using participants.tsv
            row = participants[participants['participant_id'] == sub_str]
            col = f'player{ppt}_pre_processing_channels_fixed'
            if not row.empty and col in row.columns:
                bads_str = row.iloc[0][col]
                if pd.notna(bads_str):
                    bad_list = [ch.strip() for ch in bads_str.split(',')]
                    # Verify that all bad channels exist in the epochs
                    missing = [ch for ch in bad_list if ch not in epochs.ch_names]
                    if missing:
                        logger.warning(f"Bad channels {missing} not found in channel list. Skipping those.")
                        bad_list = [ch for ch in bad_list if ch in epochs.ch_names]
                    if bad_list:
                        logger.info(f'   {sub_str} P{ppt} repairing: {bad_list}')
                        epochs = repair_bads_inverse_distance(epochs, bad_list, thresh=0.05)

            # Downsample
            epochs.resample(SFREQ)
            logger.info(f"Resampled to {SFREQ} Hz.")

            epochs.save(out_file, overwrite=True, verbose=False)
            logger.info(f"Saved: {out_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Number of pairs to process (for testing)')
    args = parser.parse_args()
    run_preprocessing(max_pairs=args.test_pairs)