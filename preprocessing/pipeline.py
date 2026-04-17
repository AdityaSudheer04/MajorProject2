import numpy as np
from preprocessing.load_data import load_dataset, clean_detection_data
from preprocessing.normalize import normalize
from preprocessing.windowing import create_windows
from preprocessing.encode_spikes import hybrid_encoding


def run_pipeline(path):

    # 1. Load
    df = load_dataset(path)

    # 2. Clean
    X, y = clean_detection_data(df)

    # 3. Normalize
    X = normalize(X)

    # 4. Create windows
    X_win, y_win = create_windows(X, y, window_size=100)

    # 5. Encode
    spikes = []
    for window in X_win:
        spikes.append(hybrid_encoding(window))

    spikes = np.array(spikes)

    return spikes, y_win