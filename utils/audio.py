"""
utils/audio.py
==============
All audio I/O and DSP — zero librosa / zero pkg_resources dependency.
Uses: soundfile, soxr (or numpy fallback), scipy, numpy
"""

import numpy as np
import soundfile as sf
import scipy.signal as sig
import scipy.interpolate as interp
import warnings
warnings.filterwarnings('ignore')

from config import (SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH,
                    N_MELS, MAX_FRAMES, EMOTION_PARAMS)

# ── Resampler (soxr preferred, numpy fallback) ────────────────────────────────
try:
    import soxr
    def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        return soxr.resample(data.astype(np.float32), sr_in, sr_out, quality='HQ')
except ImportError:
    def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return data
        n = int(len(data) * sr_out / sr_in)
        return np.interp(
            np.linspace(0, len(data) - 1, n),
            np.arange(len(data)), data
        ).astype(np.float32)


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def load_audio(path: str,
               target_sr: int = SAMPLE_RATE,
               max_seconds: float = None) -> tuple[np.ndarray, int]:
    """
    Load a WAV file as mono float32 at target_sr.
    Optionally clip to max_seconds before resampling (faster processing).
    """
    data, sr = sf.read(path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if max_seconds is not None:
        data = data[:int(sr * max_seconds)]
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
    return data.astype(np.float32), target_sr


def save_audio(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save float32 audio array as a WAV file."""
    sf.write(path, np.clip(audio, -1.0, 1.0).astype(np.float32), sr)


# ── Mel filterbank ────────────────────────────────────────────────────────────

def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a mel filterbank matrix of shape (n_mels, n_fft//2+1)."""
    hz2mel = lambda h: 2595.0 * np.log10(1.0 + h / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel_pts = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    hz_pts  = np.array([mel2hz(m) for m in mel_pts])
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb      = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, c):
            if c != lo: fb[m-1, k] = (k - lo) / (c - lo)
        for k in range(c, hi):
            if hi != c: fb[m-1, k] = (hi - k) / (hi - c)
    return fb

# Pre-compute default filterbank
_DEFAULT_FB = _build_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)


def compute_mel_spectrogram(audio: np.ndarray,
                             sr: int = SAMPLE_RATE,
                             n_fft: int = N_FFT,
                             hop: int = HOP_LENGTH,
                             n_mels: int = N_MELS) -> np.ndarray:
    """
    Compute log-mel spectrogram using scipy STFT.
    Returns array of shape (n_mels, T) in dB.
    No librosa. No pkg_resources.
    """
    window = np.hanning(n_fft).astype(np.float32)
    _, _, Zxx = sig.stft(
        audio, fs=sr, window=window,
        nperseg=n_fft, noverlap=n_fft - hop,
        boundary='constant', padded=True
    )
    power  = np.abs(Zxx) ** 2
    fb     = _DEFAULT_FB if (n_mels == N_MELS and n_fft == N_FFT and sr == SAMPLE_RATE) \
             else _build_mel_filterbank(sr, n_fft, n_mels)
    mel    = fb @ power
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
    return mel_db.astype(np.float32)


def extract_features(wav_path: str, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """
    Load WAV and return fixed-size mel feature matrix (N_MELS, max_frames).
    Used for emotion classifier training and inference.
    """
    audio, _ = load_audio(wav_path, target_sr=SAMPLE_RATE)
    mel = compute_mel_spectrogram(audio)
    if mel.shape[1] < max_frames:
        mel = np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])),
                     constant_values=-80.0)
    else:
        mel = mel[:, :max_frames]
    return mel.astype(np.float32)


# ── Emotion prosody transforms (fast numpy – no librosa) ─────────────────────

def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Pitch shift via resampling trick using numpy.interp."""
    if n_steps == 0:
        return audio
    rate   = 2.0 ** (n_steps / 12.0)
    target = max(1, int(len(audio) / rate))
    shifted = np.interp(
        np.linspace(0, len(audio) - 1, target),
        np.arange(len(audio)), audio
    )
    return np.interp(
        np.linspace(0, len(shifted) - 1, len(audio)),
        np.arange(len(shifted)), shifted
    ).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Time stretch via linear interpolation using numpy.interp."""
    if rate == 1.0:
        return audio
    target = max(1, int(len(audio) / rate))
    return np.interp(
        np.linspace(0, len(audio) - 1, target),
        np.arange(len(audio)), audio
    ).astype(np.float32)


def apply_emotion_transform(audio: np.ndarray, sr: int, emotion: str) -> np.ndarray:
    """Apply pitch, energy and speed transforms for a given emotion."""
    ps, es, ts = EMOTION_PARAMS.get(emotion, (0.0, 1.0, 1.0))
    if ps != 0:
        audio = pitch_shift(audio, sr, ps)
    audio = audio * es
    if ts != 1.0:
        audio = time_stretch(audio, ts)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
