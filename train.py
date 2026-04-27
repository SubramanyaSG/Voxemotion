"""
train.py
========
Run this script ONCE before starting the web app.
It scans the ESD dataset, extracts mel features,
trains the CNN-LSTM emotion classifier, and saves the checkpoint.

Usage:
    python train.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR, MODEL_DIR, EMOTIONS
from utils.dataset import get_dataset_df, build_training_arrays
from models.emotion_model import train_model, evaluate_model, EmotionCNNLSTM
from utils.audio import load_audio, compute_mel_spectrogram

import torch
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_mel_spectrograms(df):
    """Plot mel-spectrogram + waveform for one sample per emotion."""
    emotions = sorted(df[df['readable']]['emotion'].unique())
    n = len(emotions)
    palette = {'angry':'#ff4040','happy':'#ffd700','neutral':'#40c8ff',
               'sad':'#7a7aff','surprise':'#ff8c00'}

    fig = plt.figure(figsize=(22, 4 * n))
    fig.patch.set_facecolor('#0d0d1a')

    from config import SAMPLE_RATE
    for i, emo in enumerate(emotions):
        row   = df[(df['emotion'] == emo) & df['readable']].iloc[0]
        audio, sr = load_audio(row['file'], target_sr=SAMPLE_RATE)
        mel_db    = compute_mel_spectrogram(audio)
        clr       = palette.get(emo, '#fff')

        t_ax = np.linspace(0, len(audio) / sr, mel_db.shape[1])
        f_ax = np.linspace(0, sr / 2000, mel_db.shape[0])

        # Mel-spectrogram
        ax1 = fig.add_subplot(n, 2, i * 2 + 1)
        ax1.set_facecolor('#0d0d1a')
        im = ax1.pcolormesh(t_ax, f_ax, mel_db, cmap='magma', shading='auto')
        ax1.set_title(f'Mel-Spectrogram [{emo.upper()}]',
                      color=clr, fontsize=13, fontweight='bold')
        ax1.set_xlabel('Time (s)', color='#aaa')
        ax1.set_ylabel('Freq (kHz)', color='#aaa')
        ax1.tick_params(colors='#aaa')
        [sp.set_edgecolor(clr) for sp in ax1.spines.values()]
        plt.colorbar(im, ax=ax1, format='%+.0f dB')

        # Waveform
        ax2 = fig.add_subplot(n, 2, i * 2 + 2)
        ax2.set_facecolor('#0d0d1a')
        t_w = np.linspace(0, len(audio) / sr, len(audio))
        ax2.plot(t_w, audio, color=clr, lw=0.5, alpha=0.85)
        ax2.fill_between(t_w, audio, alpha=0.2, color=clr)
        ax2.set_title(f'Waveform [{emo.upper()}]',
                      color=clr, fontsize=13, fontweight='bold')
        ax2.set_xlabel('Time (s)', color='#aaa')
        ax2.set_ylabel('Amplitude', color='#aaa')
        ax2.tick_params(colors='#aaa')
        ax2.set_xlim([0, t_w[-1]])
        [sp.set_edgecolor(clr) for sp in ax2.spines.values()]

    fig.suptitle('Mel-Spectrograms & Waveforms by Emotion',
                 color='white', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'mel_spectrograms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f'Saved → {path}')


def plot_training_results(history, preds, trues, epochs):
    """Plot training curves and confusion matrix."""
    # Curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#0d0d1a')
    ep = range(1, epochs + 1)
    for ax, (k1, k2), ttl in [(a1, ('train_loss','val_loss'), 'Loss'),
                               (a2, ('train_acc', 'val_acc'),  'Accuracy')]:
        ax.set_facecolor('#0d0d1a')
        ax.plot(ep, history[k1], '#00e5ff', lw=2, label='Train')
        ax.plot(ep, history[k2], '#ff4081', lw=2, label='Val')
        ax.set_title(f'{ttl} Curves', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', color='#aaa')
        ax.tick_params(colors='#aaa')
        ax.legend()
        ax.grid(True, alpha=0.2, color='#444')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close(); print(f'Saved → {p}')

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(trues, preds)
    fig2, ax = plt.subplots(figsize=(8, 6))
    fig2.patch.set_facecolor('#0d0d1a'); ax.set_facecolor('#0d0d1a')
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
    ax.set_title('Confusion Matrix', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='#aaa')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close(); print(f'Saved → {p}')


def main():
    print('=' * 60)
    print('  VoxEmotion – Training Pipeline')
    print('=' * 60)
    print(f'\nDevice: {device}')

    # Step 1: Scan dataset
    print('\n[1/5] Scanning dataset …')
    df = get_dataset_df()
    if df.empty:
        print('ERROR: Dataset not found. Check DATASET_ROOT in config.py')
        sys.exit(1)

    print(f'      Total: {len(df)} files | '
          f'Readable: {df["readable"].sum()}')

    # Step 2: Plot mel spectrograms
    print('\n[2/5] Generating mel-spectrogram visualizations …')
    plot_mel_spectrograms(df)

    # Step 3: Extract features
    print('\n[3/5] Extracting mel features (up to 200 per class) …')
    X, y = build_training_arrays(df, max_per_class=200)
    print(f'      X: {X.shape}  y: {y.shape}')
    print(f'      Distribution: { {EMOTIONS[i]: int(v) for i,v in enumerate(np.bincount(y))} }')

    # Step 4: Train
    print('\n[4/5] Training CNN-LSTM emotion classifier (30 epochs) …')
    EPOCHS = 30
    model, history = train_model(X, y, epochs=EPOCHS, device=device)

    # Step 5: Evaluate and plot
    print('\n[5/5] Evaluating model …')
    EMOTION_MAP = {e: i for i, e in enumerate(EMOTIONS)}
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results = evaluate_model(model, X_te, y_te, device=device)
    plot_training_results(history, results['preds'], results['trues'], EPOCHS)

    print('\n' + '=' * 60)
    print('  Training complete!')
    print(f'  Model saved : {MODEL_DIR}/emotion_best.pth')
    print(f'  Outputs     : {OUTPUT_DIR}/')
    print('  You can now start the web app: python app.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
