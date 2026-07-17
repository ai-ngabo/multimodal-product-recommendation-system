"""
audio_pipeline.py
Task 3 - Sound Data Collection and Processing, as a runnable pipeline.

Wraps the helper functions in audio_utils.py into one class that goes
raw audio -> waveforms/spectrograms -> augmentations -> features.csv.

Usage:
    python audio_pipeline.py
or from a notebook:
    from audio_pipeline import AudioPipeline
    pipeline = AudioPipeline()
    features_df = pipeline.run()
"""

import os
import glob
import re
import pandas as pd

from audio_utils import (
    load_audio, save_wav, plot_waveform, plot_spectrogram,
    augment_pitch_shift, augment_time_stretch, augment_add_noise,
    extract_features, SAMPLE_RATE,
)


class AudioPipeline:
    """Runs the full Task 3 audio workflow end to end."""

    def __init__(
        self,
        raw_dir="../data/unrefined/audios",
        clean_dir="../data/cleaned/audios",
        features_csv="../data/cleaned/audio_features.csv",
        show_plots=False,
    ):
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        self.features_csv = features_csv
        self.show_plots = show_plots  # False = save only, don't pop up plots (useful for script mode)

        self.waveform_dir = f"{clean_dir}/waveforms"
        self.spectrogram_dir = f"{clean_dir}/spectrograms"
        self.augmented_dir = f"{clean_dir}/augmented"

    def _find_samples(self):
        """Grab every raw audio file and parse its member/sample_id from the filename."""
        paths = sorted(glob.glob(f"{self.raw_dir}/*.m4a")) + sorted(glob.glob(f"{self.raw_dir}/*.wav"))
        samples = []
        for path in paths:
            filename = os.path.basename(path)
            match = re.match(r"([a-zA-Z]+)_(\d+)\.\w+", filename)
            if not match:
                print(f"skipping {filename}, doesn't match member_number pattern")
                continue
            member, num = match.group(1).lower(), match.group(2)
            samples.append({"member": member, "sample_id": f"{member}_{num}", "path": path})
        return samples

    def visualize(self, y, sr, sample_id):
        """Save waveform + spectrogram for one sample."""
        plot_waveform(y, sr, sample_id, save_path=f"{self.waveform_dir}/{sample_id}.png")
        plot_spectrogram(y, sr, sample_id, save_path=f"{self.spectrogram_dir}/{sample_id}.png")

    def augment(self, y, sr, sample_id):
        """Create original + 3 augmented versions, save as wav, return list of (aug_name, y)."""
        versions = [
            ("original", y),
            ("pitch_shift", augment_pitch_shift(y, sr, n_steps=4)),
            ("time_stretch", augment_time_stretch(y, rate=0.85)),
            ("background_noise", augment_add_noise(y, noise_factor=0.005)),
        ]
        for aug_name, y_aug in versions:
            save_wav(y_aug, sr, f"{self.augmented_dir}/{sample_id}_{aug_name}.wav")
        return versions

    def run(self):
        """Full pipeline: load -> visualize -> augment -> extract features -> save csv."""
        samples = self._find_samples()
        print(f"found {len(samples)} raw samples in {self.raw_dir}")

        rows = []
        for sample in samples:
            y, sr = load_audio(sample["path"])
            self.visualize(y, sr, sample["sample_id"])

            versions = self.augment(y, sr, sample["sample_id"])
            for aug_name, y_aug in versions:
                feats = extract_features(y_aug, sr)
                feats["member"] = sample["member"]
                feats["sample_id"] = sample["sample_id"]
                feats["augmentation"] = aug_name
                rows.append(feats)

        features_df = pd.DataFrame(rows)
        id_cols = ["member", "sample_id", "augmentation"]
        features_df = features_df[id_cols + [c for c in features_df.columns if c not in id_cols]]

        os.makedirs(os.path.dirname(self.features_csv), exist_ok=True)
        features_df.to_csv(self.features_csv, index=False)
        print(f"saved {len(features_df)} rows to {self.features_csv}")

        return features_df


if __name__ == "__main__":
    pipeline = AudioPipeline()
    pipeline.run()
