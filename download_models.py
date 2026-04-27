# Run this as: python download_models.py
import torch
import os

print("Downloading Tacotron2...")
t2 = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_tacotron2',
    model_math='fp32',
    verbose=True
)
torch.save(t2.state_dict(), 'models/tacotron2_weights.pth')
print("Tacotron2 saved to models/tacotron2_weights.pth")

print("Downloading WaveGlow...")
wg = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_waveglow',
    model_math='fp32',
    verbose=True
)
torch.save(wg.state_dict(), 'models/waveglow_weights.pth')
print("WaveGlow saved to models/waveglow_weights.pth")