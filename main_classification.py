from preprocessing.pipeline_classification import run_classification_pipeline
import torch
from utils.visualize import compute_spike_density,plot_spike_heatmap, plot_spike_raster

spikes, labels = run_classification_pipeline("data/raw/classData.csv")

X_tensor = torch.tensor(spikes, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32)

# Rearrange for SNN
X_tensor = X_tensor.permute(1, 0, 2)

print("X shape:", X_tensor.shape)
print("y shape:", y_tensor.shape)

print(y_tensor[:5])

# Take one sample
sample = X_tensor[:, 0, :]

# Visualize
plot_spike_raster(sample)
plot_spike_heatmap(sample)

# Check sparsity
compute_spike_density(X_tensor)