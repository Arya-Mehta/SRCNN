import torch
import torch.nn as nn

# Model
class SRCNN(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.extraction = nn.Sequential(
        nn.Conv2d(in_channels=channels,
                  out_channels=64,
                  kernel_size=9,
                  padding=9//2),
        nn.ReLU()
    )
    
    self.mapping = nn.Sequential(
        nn.Conv2d(in_channels=64,
                  out_channels=32,
                  kernel_size=3,
                  padding=3 // 2),
        nn.ReLU()
    )
    
    self.reconstruction = nn.Sequential(
        nn.Conv2d(in_channels=32,
                  out_channels=channels,
                  kernel_size=5, 
                  padding=5//2),
    )

  def forward(self, X):
    out = self.extraction(X)
    out = self.mapping(out)
    out = self.reconstruction(out)
    return out
