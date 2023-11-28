import torch

seed = 42

# Device configuration - uses GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

