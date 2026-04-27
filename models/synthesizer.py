"""
models/synthesizer.py
=====================
Emotion-aware TTS synthesizer.
  Primary  : NVIDIA Tacotron2 + WaveGlow (torch.hub)
  Fallback : Retrieval-based synthesis from ESD dataset
  Emotion  : Pitch shift + energy scale + time stretch via numpy
"""

import os
import uuid
import numpy as np
import torch

from config import (OUTPUT_DIR, SAMPLE_RATE, SILENCE_SAMPLES,
                    T2_MAX_CHARS, EMOTION_PARAMS)
from utils.audio import load_audio, save_audio, apply_emotion_transform
from utils.text_utils import normalize_text, split_into_chunks


class EmotionSynthesizer:
    """
    Sentence-chunked emotion-aware TTS.
    Loads Tacotron2 once and reuses it across all synthesis calls.
    """

    WPS = 2.5   # estimated words per second for retrieval duration hint

    def __init__(self, dataset_df, output_dir: str = OUTPUT_DIR,
                 sr: int = SAMPLE_RATE):
        self.df = dataset_df[dataset_df['readable'] == True].copy() \
                  if (dataset_df is not None and not dataset_df.empty) else \
                  __import__('pandas').DataFrame()
        self.out = output_dir
        self.sr  = sr

        self._t2    = None
        self._wg    = None
        self._utils = None
        self._failed = False
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_tacotron2()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_tacotron2(self) -> bool:
        if self._failed:
            return False

        t2_weights = os.path.join(os.path.dirname(__file__), 'tacotron2_weights.pth')
        wg_weights = os.path.join(os.path.dirname(__file__), 'waveglow_weights.pth')

        # ── Strategy 1: Load from local weight files (fastest, no internet needed) ──
        if os.path.exists(t2_weights) and os.path.exists(wg_weights):
            try:
                print('⏳  Loading Tacotron2 from local weights ...')
                self._t2 = torch.hub.load(
                    'NVIDIA/DeepLearningExamples:torchhub',
                    'nvidia_tacotron2',
                    model_math='fp32',
                    verbose=False,
                    pretrained=False          # skip weight download
                ).to(self._device).eval()
                self._t2.load_state_dict(
                    torch.load(t2_weights, map_location=self._device)
                )

                self._wg = torch.hub.load(
                    'NVIDIA/DeepLearningExamples:torchhub',
                    'nvidia_waveglow',
                    model_math='fp32',
                    verbose=False,
                    pretrained=False          # skip weight download
                ).to(self._device).eval()
                self._wg.load_state_dict(
                    torch.load(wg_weights, map_location=self._device)
                )

                self._utils = torch.hub.load(
                    'NVIDIA/DeepLearningExamples:torchhub',
                    'nvidia_tts_utils',
                    verbose=False
                )
                for m in self._wg.modules():
                    if hasattr(m, 'weight_v'):
                        torch.nn.utils.remove_weight_norm(m)

                print('✅  Tacotron2 loaded from local weights.')
                return True
            except Exception as e:
                print(f'⚠  Local weight load failed: {e}')

        # ── Strategy 2: Download from torch.hub (needs internet, may rate-limit) ──
        try:
            print('⏳  Downloading Tacotron2 from torch.hub ...')
            self._t2 = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub',
                'nvidia_tacotron2',
                model_math='fp32',
                verbose=False
            ).to(self._device).eval()

            self._wg = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub',
                'nvidia_waveglow',
                model_math='fp32',
                verbose=False
            ).to(self._device).eval()

            self._utils = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub',
                'nvidia_tts_utils',
                verbose=False
            )
            for m in self._wg.modules():
                if hasattr(m, 'weight_v'):
                    torch.nn.utils.remove_weight_norm(m)

            # Save locally so next restart doesn't need to download
            try:
                os.makedirs(os.path.dirname(t2_weights), exist_ok=True)
                torch.save(self._t2.state_dict(), t2_weights)
                torch.save(self._wg.state_dict(), wg_weights)
                print('✅  Weights saved locally for future restarts.')
            except Exception:
                pass

            print('✅  Tacotron2 downloaded and loaded.')
            return True

        except Exception as e:
            print(f'⚠  Tacotron2 not available ({e})')
            print('   → Retrieval-based synthesis will be used.')
            self._failed = True
            return False

    # ── Synthesis methods ─────────────────────────────────────────────────────
    def _synthesize_chunk_t2(self, chunk: str) -> np.ndarray:
        """Synthesize one short text chunk with Tacotron2."""
        seqs, lengths = self._utils.prepare_input_sequence([chunk])
        with torch.no_grad():
            mel, _, _ = self._t2.infer(
                seqs.to(self._device), lengths.to(self._device)
            )
            audio = self._wg.infer(mel)
        return audio[0].cpu().numpy().astype(np.float32)

    def _retrieval_chunk(self, emotion: str, dur_hint: float = 3.0) -> np.ndarray:
        """Load a matching ESD audio clip trimmed to dur_hint seconds."""
        sub = self.df[self.df['emotion'] == emotion]
        if sub.empty:
            sub = self.df
        if sub.empty:
            return np.zeros(int(self.sr * dur_hint), dtype=np.float32)
        try:
            audio, _ = load_audio(
                sub.sample(1).iloc[0]['file'],
                target_sr=self.sr,
                max_seconds=dur_hint
            )
            needed = int(self.sr * dur_hint)
            if len(audio) < needed:
                audio = np.pad(audio, (0, needed - len(audio)))
            return audio
        except Exception:
            return np.zeros(int(self.sr * dur_hint), dtype=np.float32)

    # ── Main synthesis entry point ────────────────────────────────────────────
    def synthesize(self, text: str, emotion: str = 'neutral',
                   filename: str = None) -> dict:
        """
        Synthesize speech for any-length text with emotion control.

        Flow:
          1. Normalize text (remove special chars, expand numbers)
          2. Split into sentence chunks ≤ T2_MAX_CHARS each
          3. Synthesize each chunk (Tacotron2 or retrieval)
          4. Concatenate chunks with 180ms silence gaps
          5. Apply emotion prosody transform (pitch/energy/speed)
          6. Save WAV and return metadata dict

        Returns:
          {'file': filename, 'text': text, 'emotion': emotion, 'duration': float}
        """
        emotion = emotion.lower() if emotion.lower() in EMOTION_PARAMS else 'neutral'
        text    = normalize_text(text)[:10_000_000]
        cks     = split_into_chunks(text, max_chars=T2_MAX_CHARS)
        if not cks:
            cks = [text[:T2_MAX_CHARS]]

        silence = np.zeros(SILENCE_SAMPLES, dtype=np.float32)
        parts   = []
        sr      = 22050 if self.using_tacotron2 else self.sr

        print(f'  Synthesizing {len(cks)} chunk(s) [{emotion}]')

        for i, chunk in enumerate(cks):
            chunk = chunk.strip()
            if not chunk:
                continue
            seg = None

            if self.using_tacotron2:
                try:
                    seg = self._synthesize_chunk_t2(chunk)
                except Exception as e:
                    print(f'  Chunk {i+1} Tacotron2 failed: {e} → using retrieval')

            if seg is None:
                dur_hint = max(2.0, len(chunk.split()) / self.WPS)
                seg      = self._retrieval_chunk(emotion, dur_hint)
                sr       = self.sr

            parts.append(seg)
            if i < len(cks) - 1:
                parts.append(silence)

        if not parts:
            audio = np.zeros(sr * 3, dtype=np.float32)
        else:
            audio = np.concatenate(parts).astype(np.float32)

        # Apply emotion transform on complete audio
        audio = apply_emotion_transform(audio, sr, emotion)

        # Save
        fname    = filename or f'tts_{emotion}_{uuid.uuid4().hex[:8]}.wav'
        out_path = os.path.join(self.out, fname)
        save_audio(out_path, audio, sr)

        duration = round(len(audio) / sr, 2)
        print(f'  ✅  {duration:.1f}s saved → {fname}')

        return {
            'file'    : fname,
            'text'    : text,
            'emotion' : emotion,
            'duration': duration
        }
