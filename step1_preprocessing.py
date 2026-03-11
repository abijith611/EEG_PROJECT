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

path_to_data = 'project/ds006761'
pair_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

def repair_bads_inverse_distance(epochs, bad_chans, thresh=0.05):
    """
    Replicate MATLAB's ft_channelrepair with method='distance'.
    For each bad channel, find neighbours within `thresh` meters,
    compute inverse‑distance weights, and replace the bad channel's data
    with the weighted average of its neighbours.
    If no neighbours are found within the threshold, fall back to using all
    good channels (with inverse‑distance weighting based on all distances).

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object with bad channels marked in `bad_chans`.
    bad_chans : list of str
        Names of channels to repair (must match epochs.ch_names).
    thresh : float
        Distance threshold in meters (default 0.05 = 5 cm).

    Returns
    -------
    epochs : mne.Epochs
        The same epochs object with data repaired (modified in‑place).
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
            print(f"Warning: channel {bad} not found, skipping.")
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
            # Avoid division by zero for the bad channel itself (excluded)
            dists = dist_mat[bad_idx, neigh_idx]
        else:
            dists = dist_mat[bad_idx, neigh_idx]

        # --- Print repair info ---
        print(f"repairing channel {bad}")
        for nb_idx in neigh_idx:
            print(f"    using neighbour {ch_names[nb_idx]}")
        # -------------------------

        # Compute inverse‑distance weights
        with np.errstate(divide='ignore'):
            weights = 1.0 / dists
        weights[np.isinf(weights)] = 0  # in case of zero distance (shouldn't happen)
        weights /= weights.sum()

        # Weighted average across neighbours for all time points and epochs
        # data[:, neigh_idx, :] shape: (n_epochs, n_neigh, n_times)
        data[:, bad_idx, :] = np.average(data[:, neigh_idx, :], axis=1, weights=weights)

    # After processing all bad channels, print total trials
    print(f"interpolating bad channels for {len(epochs)} trials.")

    # Update epochs data in‑place (preserves montage, events, etc.)
    epochs._data = data
    return epochs

def run_preprocessing(max_pairs=None):
    deriv_dir = os.path.join(path_to_data, 'derivatives')
    os.makedirs(deriv_dir, exist_ok=True)

    participants = pd.read_csv(os.path.join(path_to_data, 'participants.tsv'), sep='\t')
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids

    for pair in pairs_to_run:
        sub_str = f'sub-{pair:02d}'
        eeg_dir = os.path.join(path_to_data, sub_str, 'eeg')

        raw = mne.io.read_raw_bdf(os.path.join(eeg_dir, f'{sub_str}_task-RPS_eeg.bdf'),
                                  preload=True, verbose=False)
        events_df = pd.read_csv(os.path.join(eeg_dir, f'{sub_str}_task-RPS_events.tsv'),
                                sep='\t')

        montage = mne.channels.make_standard_montage('biosemi64')

        for ppt in [1, 2]:
            out_file = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS-epo.fif')
            if os.path.exists(out_file):
                print(f"Skipping pair {pair} player {ppt} – output already exists.")
                continue
            prefix = '2-' if ppt == 1 else '1-'
            # First, pick only the EEG channels
            ppt_chans = [ch for ch in raw.ch_names if (ch.startswith(prefix + 'A') or ch.startswith(prefix + 'B'))]
            raw_ppt = raw.copy().pick(ppt_chans)

            # The exact label list extracted from the author's FieldTrip layout
            matlab_layout_labels = [
                'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 
                'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 
                'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 
                'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 
                'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 
                'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 
                'P10', 'PO8', 'PO4', 'O2'
            ]

            # Force the MATLAB channel scrambling: map current hardware channels sequentially
            current_chans = raw_ppt.ch_names
            forced_mapping = {current_chans[i]: matlab_layout_labels[i] for i in range(len(current_chans))}
            
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

            # Repair bad channels using participants.tsv
            row = participants[participants['participant_id'] == sub_str]
            col = f'player{ppt}_pre_processing_channels_fixed'
            if not row.empty and col in row.columns:
                bads_str = row.iloc[0][col]
                if pd.notna(bads_str):
                    bad_list = [ch.strip() for ch in bads_str.split(',')]
                    print(f'   {sub_str} P{ppt} repairing: {bad_list}')
                    epochs = repair_bads_inverse_distance(epochs, bad_list, thresh=0.05)

            # Downsample
            epochs.resample(256.0)

            epochs.save(out_file, overwrite=True, verbose=False)

if __name__ == '__main__':
    # Add command‑line argument for testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Number of pairs to process (for testing)')
    args = parser.parse_args()
    run_preprocessing(max_pairs=args.test_pairs)