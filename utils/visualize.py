import matplotlib.pyplot as plt
import numpy as np


# Spike Raster Plot
def plot_spike_raster(sample, title="Spike Raster Plot"):
    """
    sample: (time_steps, features)
    """

    sample = sample.cpu().numpy() if hasattr(sample, "cpu") else sample

    time_steps, features = sample.shape

    plt.figure(figsize=(10, 5))

    for f in range(features):
        for t in range(time_steps):
            if sample[t, f] == 1:
                plt.scatter(t, f, color='black', s=10)

    plt.xlabel("Time step")
    plt.ylabel("Feature index")
    plt.title(title)
    plt.yticks(range(features))
    plt.grid(alpha=0.3)
    plt.show()



# Spike Heatmap
def plot_spike_heatmap(sample, title="Spike Heatmap"):
    """
    sample: (time_steps, features)
    """

    sample = sample.cpu().numpy() if hasattr(sample, "cpu") else sample

    plt.figure(figsize=(10, 5))
    plt.imshow(sample.T, aspect='auto', cmap='gray_r')
    plt.colorbar(label="Spike")
    plt.xlabel("Time step")
    plt.ylabel("Feature index")
    plt.title(title)
    plt.show()


# Raw Signal Plot
def plot_raw_signal(sample, title="Raw Signal"):
    """
    sample: (time_steps, features)
    """

    sample = sample.cpu().numpy() if hasattr(sample, "cpu") else sample

    plt.figure(figsize=(10, 5))

    for f in range(sample.shape[1]):
        plt.plot(sample[:, f], label=f"Feature {f}")

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# Spike Density Check
def compute_spike_density(spikes):
    """
    spikes: full dataset (time, samples, features)
    """

    spikes_np = spikes.cpu().numpy() if hasattr(spikes, "cpu") else spikes

    total_elements = spikes_np.size
    total_spikes = np.sum(spikes_np)

    density = total_spikes / total_elements

    print(f"Spike density: {density:.4f}")

    return density



# Compare Two Samples
def compare_samples(sample1, sample2):
    """
    Compare two spike samples side by side
    """

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(sample1.T, aspect='auto', cmap='gray_r')
    plt.title("Sample 1")

    plt.subplot(1, 2, 2)
    plt.imshow(sample2.T, aspect='auto', cmap='gray_r')
    plt.title("Sample 2")

    plt.show()