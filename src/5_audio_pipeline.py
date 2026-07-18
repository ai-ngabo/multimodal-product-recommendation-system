"""Stage 5: Sound data collection & processing.

Member voice recordings are read from data/unrefined/audios/, normalised to a
canonical form (mono -> resample -> silence-trimmed -> loudness-normalised ->
fixed duration), cached to data/cleaned/audios/, then augmented (>=2 per
recording) and reduced to a feature vector.

Steps: ingest -> normalise -> waveform+spectrogram -> audio augmentations -> features(csv)

Features: 13 MFCC means + 13 MFCC stds + spectral roll-off + RMS energy + ZCR
          -> outputs/features/audio_features.csv 
"""
import hashlib

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (RAW_AUDIO, CLEAN_AUDIO, PLOTS, FEATURES, USERS, PHRASES,
                    SR, DURATION, SEED, ROOT)

AUDIO_FORMAT = (".wav", ".m4a", ".mp3", ".flac", ".ogg")
META_COLS = ["member", "phrase", "variant", "source"]

N_MFCC = 13
TRIM_DB = 25   # anything this far below peak is treated as silence
MIN_SPEECH_SEC = 0.3  # below this after trimming, the capture is suspect



# ingest & locating audio files
def find_source(user, phrase):
    """Locate data/unrefined/audios/<user>_<phrase>.<ext>.

    Case- and extension-insensitive, mirroring find_source() in stage 4.
    """
    stem = f"{user}_{phrase}".lower()
    if not RAW_AUDIO.is_dir():
        raise FileNotFoundError(f"Raw audio directory not found: {RAW_AUDIO}")
    for f in sorted(RAW_AUDIO.iterdir()):
        if f.suffix.lower() in AUDIO_FORMAT and f.stem.lower() == stem:
            return f
    raise FileNotFoundError(
        f"No recording for {user}/{phrase}. Expected "
        f"{RAW_AUDIO}/{user}_{phrase}.wav (or .m4a/.mp3/.flac/.ogg)")

# audio normalizing function
def normalise(path):
    """Raw recording -> (canonical mono float array at SR, speech_found).

    Four corrections, each addressing a real property of phone recordings:
      1. mono + resample  -- devices record at 44.1k/48k, stereo or mono
      2. trim silence     -- leading/trailing dead air is device habit, not voice
      3. RMS normalise    -- removes recording-level differences between devices
      4. fixed duration   -- equal-length clips so energy features are comparable
    """
    y, _ = librosa.load(path, sr=SR, mono=True)

    y_trim, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    speech_found = len(y_trim) >= MIN_SPEECH_SEC * SR
    if not speech_found:
        y_trim = y  # keep the original audio

    rms = float(np.sqrt(np.mean(y_trim ** 2)))
    if rms > 1e-6:
        y_trim = y_trim * (0.1 / rms)    # target RMS
    y_trim = np.clip(y_trim, -1.0, 1.0)

    target = int(DURATION * SR)
    if len(y_trim) < target:
        y_trim = np.pad(y_trim, (0, target - len(y_trim)))
    else:
        y_trim = y_trim[:target]

    return y_trim.astype(np.float32), speech_found


# audio augmentation + features

def _seed_from(key):
    """Stable seed from a string. hashlib, not hash() - Python salts string
    hashing per process, which would make runs irreproducible."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def augment(y, key):
    """ augmentations, deterministic per source recording.

    Pitch shift is deliberately small (+/-1 semitone). Pitch is a primary
    speaker-identity cue, so a large shift would generate samples labelled with
    one member that sound like another -- label noise, not augmentation. One
    semitone simulates natural day-to-day vocal variation instead.
    """
    r = np.random.default_rng(_seed_from(key) ^ SEED)
    steps = float(r.uniform(-1.0, 1.0))
    rate = float(r.uniform(0.9, 1.1))
    noise = r.normal(0, 0.005, len(y)).astype(np.float32)

    stretched = librosa.effects.time_stretch(y, rate=rate)
    target = int(DURATION * SR)
    if len(stretched) < target:
        stretched = np.pad(stretched, (0, target - len(stretched)))
    else:
        stretched = stretched[:target]

    return {
        "pitch": librosa.effects.pitch_shift(y, sr=SR, n_steps=steps),
        "stretch": stretched,
        "noise": np.clip(y + noise, -1.0, 1.0),
        "quiet": y * 0.6,
    }


def features(y):
    """13 MFCC means + 13 MFCC stds + roll-off + RMS energy + ZCR (29 dims).

    Frame-level descriptors are aggregated to mean/std so the vector is a fixed
    length regardless of clip duration. MFCC stds matter as much as means: they
    capture how much a speaker's timbre moves during the phrase.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        [rolloff.mean()], [rms.mean()], [zcr.mean()],
    ])



# CLI entry point for audio pipeline
def main():
    for d in (CLEAN_AUDIO, PLOTS / "audio_pipeline"):
        d.mkdir(parents=True, exist_ok=True)

    rows, grid = [], []

    for m in USERS:
        for p in PHRASES:
            src = find_source(m, p)
            y, speech_found = normalise(src)
            sf.write(CLEAN_AUDIO / f"{m}_{p}.wav", y, SR)   # canonical copy
            grid.append((f"{m}\n{p}", y, speech_found))

            variants = {"orig": y, **augment(y, f"{m}_{p}")}
            for vname, vy in variants.items():
                rows.append({"member": m, "phrase": p, "variant": vname,
                             "source": src.name,
                             **{f"a{i}": v for i, v in enumerate(features(vy))}})

    # waveform + spectrogram grid (doubles as ingestion check) 
    fig = plt.figure(figsize=(14, 2.2 * len(grid)))
    gs = gridspec.GridSpec(len(grid), 2, figure=fig, hspace=0.7, wspace=0.2)
    for i, (title, y, ok) in enumerate(grid):
        ax = fig.add_subplot(gs[i, 0])
        librosa.display.waveshow(y, sr=SR, ax=ax)
        ax.set_title(f"{title.replace(chr(10), ' / ')} — waveform",
                     fontsize=9, color="black" if ok else "red")
        ax.set_xlabel("")

        ax = fig.add_subplot(gs[i, 1])
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(f"{title.replace(chr(10), ' / ')} — spectrogram", fontsize=9)
        ax.set_xlabel("")
    fig.suptitle("Voice captures — 4 users × 2 phrases (normalised)\n"
                 "red = little speech detected after silence trimming",
                 y=1.005, fontsize=13)
    fig.savefig(PLOTS / "audio_pipeline" / "audio_wave_spectrogram.png",
                dpi=110, bbox_inches="tight")
    plt.close(fig)

    # augmentations for one recording 
    demo, _ = normalise(find_source(USERS[0], PHRASES[0]))
    augs = {"original": demo, **augment(demo, f"{USERS[0]}_{PHRASES[0]}")}
    fig, axes = plt.subplots(len(augs), 1, figsize=(11, 2.0 * len(augs)),
                             sharex=True)
    for ax, (name, ya) in zip(axes, augs.items()):
        librosa.display.waveshow(ya, sr=SR, ax=ax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("")
    fig.suptitle(f"Audio augmentations ({USERS[0]} / {PHRASES[0]})", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "audio_pipeline" / "audio_augmentations.png",
                dpi=110, bbox_inches="tight")
    plt.close(fig)

    # save the features
    df = pd.DataFrame(rows)
    df.to_csv(FEATURES / "audio_features.csv", index=False)
    df.to_csv(ROOT / "audio_features.csv", index=False) 

    # validation checks 
    n_expected = len(USERS) * len(PHRASES)
    n_speech = sum(ok for _, _, ok in grid)
    per_member = df.groupby("member").size().to_dict()

    print(f"[Done] audio_features.csv -> {df.shape[0]} rows, {df.shape[1]} cols "
          f"({df.shape[1] - len(META_COLS)} feature dims)")
    print(f"[CHECK] recordings loaded : {len(grid)}/{n_expected}")
    print(f"[CHECK] speech detected   : {n_speech}/{len(grid)} "
          f"({n_speech / len(grid):.0%}) — rest kept untrimmed, inspect the waveforms")
    print(f"[CHECK] rows per member   : {per_member}")

    assert len(grid) == n_expected, "missing recordings — see errors above"
    assert df.groupby("member").size().nunique() == 1, \
        f"class imbalance across members: {per_member}"
    assert not df.filter(like="a").isna().any().any(), "NaN in feature columns"
    print("[Done] Quality Control assertions passed.")


if __name__ == "__main__":
    main()