"""
models/emotion_model.py
=======================
CNN-LSTM emotion classifier.
Input:  mel-spectrogram image (1, N_MELS, MAX_FRAMES)
Output: emotion class (0-4 mapping to EMOTIONS list)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config import (MODEL_DIR, OUTPUT_DIR, EMOTIONS, N_MELS, MAX_FRAMES)


# ── Dataset wrapper ───────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,80,300)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── CNN-LSTM Model ────────────────────────────────────────────────────────────
class EmotionCNNLSTM(nn.Module):
    """
    Architecture:
      CNN encoder  : 3x Conv2D blocks → spatial features
      BiLSTM       : 2-layer bidirectional → temporal patterns
      Classifier   : FC layers → emotion classes
    """
    def __init__(self, n_classes: int = 5,
                 n_mels: int = N_MELS,
                 frames: int = MAX_FRAMES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        # After 3x MaxPool(2): height = n_mels//8
        lstm_input = 128 * (n_mels // 8)
        self.lstm = nn.LSTM(
            input_size   = lstm_input,
            hidden_size  = 256,
            num_layers   = 2,
            batch_first  = True,
            dropout      = 0.3,
            bidirectional= True,
        )
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 80, 300)
        x = self.cnn(x)                        # (B, 128, H', W')
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2)             # (B, W', C, H')
        x = x.reshape(B, W, C * H)            # (B, W', C*H')
        x, _ = self.lstm(x)                   # (B, W', 512)
        return self.head(x[:, -1, :])         # last timestep


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(X: np.ndarray, y: np.ndarray,
                epochs: int = 30,
                batch_size: int = 32,
                device: str = None) -> tuple[EmotionCNNLSTM, dict]:
    """
    Train the emotion classifier.
    Returns (trained_model, training_history).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tr_dl = DataLoader(EmotionDataset(X_tr, y_tr),
                       batch_size=batch_size, shuffle=True, num_workers=0)
    te_dl = DataLoader(EmotionDataset(X_te, y_te),
                       batch_size=batch_size, shuffle=False, num_workers=0)

    model     = EmotionCNNLSTM(n_classes=len(EMOTIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history  = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_acc = 0.0
    ckpt     = os.path.join(MODEL_DIR, 'emotion_best.pth')

    print(f'Training on {len(X_tr)} | Val on {len(X_te)} | Device: {device}\n')

    for ep in range(1, epochs + 1):
        model.train(); tl = tc = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(yb)
            tc += (out.argmax(1) == yb).sum().item()
        scheduler.step()

        model.eval(); vl = vc = 0
        with torch.no_grad():
            for xb, yb in te_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += criterion(out, yb).item() * len(yb)
                vc += (out.argmax(1) == yb).sum().item()

        ta = tc / len(X_tr); va = vc / len(X_te)
        history['train_loss'].append(tl / len(X_tr))
        history['val_loss'].append(vl / len(X_te))
        history['train_acc'].append(ta)
        history['val_acc'].append(va)

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), ckpt)

        if ep % 5 == 0 or ep == 1:
            print(f'Epoch {ep:3d}  TrLoss {tl/len(X_tr):.4f}  '
                  f'TrAcc {ta:.4f}  VaLoss {vl/len(X_te):.4f}  VaAcc {va:.4f}')

    print(f'\n🏆  Best Val Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, history


def load_model(device: str = None) -> EmotionCNNLSTM | None:
    """Load saved emotion classifier from checkpoint."""
    ckpt = os.path.join(MODEL_DIR, 'emotion_best.pth')
    if not os.path.exists(ckpt):
        return None
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EmotionCNNLSTM(n_classes=len(EMOTIONS)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def evaluate_model(model: EmotionCNNLSTM, X_te: np.ndarray,
                   y_te: np.ndarray, device: str = None) -> dict:
    """Run evaluation and return classification report dict."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    te_dl = DataLoader(EmotionDataset(X_te, y_te), batch_size=32)
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
            trues.extend(yb.tolist())
    print(classification_report(trues, preds, target_names=EMOTIONS, digits=4))
    return {'preds': preds, 'trues': trues,
            'cm': confusion_matrix(trues, preds)}
