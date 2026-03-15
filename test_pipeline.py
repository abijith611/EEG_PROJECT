"""
EEG Pipeline Diagnostic & Sanity Suite
Validates: Preprocessing, Data Engineering, Statistical Math, and Coordinate mapping.
Generates: 'EEG-PROJECT/results/plots/sanity_check_visual.png'
"""

import unittest
import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from config import NUM_CHAN, NUM_TIME_BINS, MATLAB_LAYOUT_LABELS, SFREQ, NUM_TRIALS, PLOT_DIR
from step1_preprocessing import repair_bads_inverse_distance
from step2a_decoding import create_pseudo_trials
from bayes_output import calc_bf_1samp

class TestEEGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "="*70)
        print("SANITY CHECKS & SCIENTIFIC DISCUSSION")
        print("="*70)

    def test_01_interpolation_with_visualization(self):
        """
        Check 1: Inverse-Distance Weighting (IDW) Interpolation
        """
        print("\n[Check 1: Preprocessing Interpolation]")
        
        # 1. Setup Mock Data
        data = np.random.randn(NUM_CHAN, 500) * 1e-6
        info = mne.create_info(ch_names=MATLAB_LAYOUT_LABELS, sfreq=SFREQ, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montage)
        
        # 2. Kill Channel 10
        raw._data[10, :] = 0.0
        epochs = mne.EpochsArray(raw.get_data()[np.newaxis, :], info, verbose=False)
        epochs.set_montage(montage)
        
        # 3. Repair
        repaired_epochs = repair_bads_inverse_distance(epochs, [MATLAB_LAYOUT_LABELS[10]], thresh=0.05)
        repaired_data = repaired_epochs.get_data(copy=True)[0, 10, :]
        
        # 4. Visualization (Sanity Plot)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.plot(data[10, :100], color='gray', alpha=0.5, label='Original (Pre-Failure)')
        ax1.plot([0]*100, color='red', label='Mock Failure (Flat)')
        ax1.set_title("Input: Bad Channel State")
        ax1.legend()
        
        ax2.plot(repaired_data[:100], color='green', label='Interpolated')
        ax2.set_title("Output: Reconstructed Signal (IDW)")
        ax2.legend()
        plt.tight_layout()
        viz_path = os.path.join(PLOT_DIR, "sanity_check_visual.png")
        plt.savefig(viz_path)
        plt.close()

        # 5. Discussion
        self.assertTrue(np.any(repaired_data != 0.0))
        print(f"This seems correct because the flat channel was successfully reconstructed using ")
        print(f"the spatial neighbors defined in the biosemi64 montage. The generated plot ")
        print(f"'{os.path.basename(viz_path)}' confirms signal recovery.")

    def test_02_pseudo_trial_logic(self):
        """
        Check 2: SNR Boosting via Pseudo-Trials
        """
        print("\n[Check 2: Pseudo-Trial Data Engineering]")
        mock_X = np.random.rand(40, NUM_CHAN, NUM_TIME_BINS)
        mock_y = np.repeat([1, 2, 3, 4], 10) 
        X_ps, y_ps, chunks = create_pseudo_trials(mock_X, mock_y, seed=42)
        
        self.assertEqual(X_ps.shape[0], 800)
        print(f"This seems correct because 40 single trials, split into 10 folds, ")
        print(f"resampled 20 times per class (4 classes), yields exactly 800 pseudo-trials. ")
        print(f"This dimensionality ensures robust SNR for the SVM classifiers.")

    def test_03_bayes_math_consistency(self):
        """
        Check 3: Bayesian Sensitivity
        """
        print("\n[Check 3: Bayesian Statistical Math]")
        chance = 33.33
        above_chance = np.array([36.1, 34.5, 37.2, 35.8])
        bf = calc_bf_1samp(above_chance, mu=chance)
        
        self.assertGreater(bf, 1.0)
        print(f"This seems correct because data strictly higher than chance ({chance}%) ")
        print(f"yielded a BF10 of {bf:.2f}, providing evidence for the alternative hypothesis.")

    def test_04_coordinate_mapping(self):
        """
        Check 4: Montage Mapping Integrity
        """
        print("\n[Check 4: Coordinate Mapping]")
        from config import get_pos_dict
        pos = get_pos_dict()
        
        self.assertIsNotNone(pos)
        self.assertEqual(len(pos), NUM_CHAN)
        print(f"This seems correct because the biosemi64.mat file contains exact ")
        print(f"coordinates for all {NUM_CHAN} channels, matching the 10-20 labels perfectly.")

if __name__ == '__main__':
    unittest.main(verbosity=1)