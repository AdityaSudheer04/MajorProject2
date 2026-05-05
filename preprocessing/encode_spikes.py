import numpy as np


# --- Delta Encoding ---
def delta_encoding(window, threshold=0.05):

    spikes = np.zeros_like(window)

    for t in range(1, window.shape[0]):
        diff = window[t] - window[t-1]
        spikes[t] = (diff > threshold).astype(int)

    return spikes


# --- Latency Encoding ---
def latency_encoding(window, threshold=0.3):  # ✅ Lowered from 0.7 → 0.3
    # Why: threshold=0.7 was too aggressive — weaker fault types (LL, LLG) produce
    # lower-amplitude signals that were being silenced, starving the classifier.
    # 0.3 preserves fault-discriminative information across all 4 fault classes.

    T = window.shape[0]
    encoded = np.zeros_like(window)

    for f in range(window.shape[1]):

        values = window[:, f]

        v_min, v_max = values.min(), values.max()
        norm = (values - v_min) / (v_max - v_min + 1e-8)

        spike_times = (1 - norm) * (T - 1)
        spike_times = spike_times.astype(int)

        for t, st in enumerate(spike_times):
            if norm[t] > threshold:   # ✅ More inclusive — catches weaker fault signals
                encoded[st, f] = 1

    return encoded


# --- Hybrid Encoding ---
def hybrid_encoding(window):

    delta_spikes = delta_encoding(window)
    latency_spikes = latency_encoding(window)

    # combine but avoid flooding
    combined = delta_spikes.copy()
    combined[latency_spikes == 1] = 1

    return combined