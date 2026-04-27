"""
utils/dataset.py
================
ESD dataset scanning, metadata building, and feature extraction
for the CNN-LSTM emotion classifier.
"""

import os
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config import (DATASET_ROOT, OUTPUT_DIR, EMOTIONS,
                    SAMPLE_RATE, MAX_FRAMES)
from utils.audio import load_audio, compute_mel_spectrogram

# ── Cached dataset DataFrame ──────────────────────────────────────────────────
_df_cache: pd.DataFrame | None = None


def get_dataset_df(force_rescan: bool = False) -> pd.DataFrame:
    """
    Return dataset metadata DataFrame.
    Loads from CSV cache if available, otherwise scans dataset directory.
    """
    global _df_cache
    if _df_cache is not None and not force_rescan:
        return _df_cache

    csv_path = os.path.join(OUTPUT_DIR, 'dataset_metadata.csv')
    if os.path.exists(csv_path) and not force_rescan:
        _df_cache = pd.read_csv(csv_path)
        print(f'Loaded dataset metadata from {csv_path}')
        return _df_cache

    _df_cache = scan_dataset(DATASET_ROOT)
    _df_cache.to_csv(csv_path, index=False)
    print(f'Dataset metadata saved to {csv_path}')
    return _df_cache


def parse_speaker_txt(txt_path: str) -> dict:
    """
    Parse ESD speaker transcript file.
    Format: filename TAB text TAB emotion
    Returns: {stem: (text, emotion)}
    """
    mapping = {}
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    fname   = parts[0].strip()
                    text    = parts[1].strip()
                    emotion = parts[2].strip().lower() if len(parts) > 2 else ''
                    mapping[fname] = (text, emotion)
    except Exception as e:
        print(f'  ⚠  Could not parse {txt_path}: {e}')
    return mapping


def scan_dataset(root: str) -> pd.DataFrame:
    """
    Walk the ESD directory and collect all audio file metadata.
    Verifies readability using soundfile.info().
    """
    records = []
    root_path = Path(root)

    if not root_path.exists():
        print(f'⚠  Dataset root not found: {root}')
        print('   Returning empty DataFrame.')
        return pd.DataFrame(columns=['speaker','emotion','file','filename',
                                     'text','duration','readable'])

    speaker_dirs = sorted(d for d in root_path.iterdir() if d.is_dir())
    print(f'Found {len(speaker_dirs)} speaker folder(s)')

    for spk_dir in tqdm(speaker_dirs, desc='Scanning dataset'):
        spk_id  = spk_dir.name
        txt_map = parse_speaker_txt(str(spk_dir / f'{spk_id}.txt'))

        for emo_dir in sorted(d for d in spk_dir.iterdir() if d.is_dir()):
            emotion = emo_dir.name.lower()
            for wav in sorted(emo_dir.glob('*.wav')):
                text, _ = txt_map.get(wav.stem, ('', emotion))
                readable, duration = False, 0.0
                try:
                    info     = sf.info(str(wav))
                    duration = info.duration
                    readable = True
                except Exception:
                    pass

                records.append({
                    'speaker' : spk_id,
                    'emotion' : emotion,
                    'file'    : str(wav),
                    'filename': wav.name,
                    'text'    : text,
                    'duration': round(duration, 3),
                    'readable': readable
                })

    df = pd.DataFrame(records)
    total = len(df)
    ok    = int(df['readable'].sum())
    print(f'\nDataset: {total} files | ✅ Readable: {ok} | ❌ Errors: {total - ok}')
    print(f'Total duration: {df["duration"].sum() / 60:.1f} minutes')
    return df


# ── Feature extraction for emotion classifier ─────────────────────────────────
EMOTION_MAP = {e: i for i, e in enumerate(EMOTIONS)}


def extract_features_for_training(wav_path: str) -> np.ndarray:
    """
    Load audio and return (N_MELS, MAX_FRAMES) mel feature matrix.
    Pads or truncates to MAX_FRAMES.
    """
    audio, _ = load_audio(wav_path, target_sr=SAMPLE_RATE)
    mel = compute_mel_spectrogram(audio)
    if mel.shape[1] < MAX_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, MAX_FRAMES - mel.shape[1])),
                     constant_values=-80.0)
    else:
        mel = mel[:, :MAX_FRAMES]
    return mel.astype(np.float32)


def build_training_arrays(df: pd.DataFrame,
                           max_per_class: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract mel features for all audio files.
    Returns X of shape (N, N_MELS, MAX_FRAMES) and y of shape (N,).
    """
    X, Y = [], []
    readable_df = df[df['readable'] & df['emotion'].isin(EMOTIONS)]

    for emotion in EMOTIONS:
        subset = readable_df[readable_df['emotion'] == emotion].head(max_per_class)
        print(f'  {emotion:10s}: {len(subset)} files')
        for _, row in tqdm(subset.iterrows(), total=len(subset),
                           desc=f'  Extracting {emotion}', leave=False):
            try:
                feat = extract_features_for_training(row['file'])
                X.append(feat)
                Y.append(EMOTION_MAP[emotion])
            except Exception as e:
                pass

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)
