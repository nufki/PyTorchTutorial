import torch
x = torch.rand(5, 3)
print(x)


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")