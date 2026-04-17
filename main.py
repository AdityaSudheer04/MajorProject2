from preprocessing.pipeline import run_pipeline
import torch
from utils.visualize import plot_spike_raster, plot_spike_heatmap, compute_spike_density

# 1. Run preprocessing pipeline
spikes, labels = run_pipeline("data/raw/detect_dataset.csv")

# 2. Convert to tensors
X_tensor = torch.tensor(spikes, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

X_tensor = X_tensor.permute(1, 0, 2)

# 3. Check shapes
print("X shape:", X_tensor.shape)
print("y shape:", y_tensor.shape)

print(y_tensor[:5])
print("Detection label distribution:")
print(y_tensor.unique(return_counts=True))

# Take one sample
sample = X_tensor[:, 0, :]

# Visualize
plot_spike_raster(sample)
plot_spike_heatmap(sample)

# Check sparsity
compute_spike_density(X_tensor)