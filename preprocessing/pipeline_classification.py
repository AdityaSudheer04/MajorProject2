import numpy as np
from preprocessing.load_data import load_dataset, clean_classification_data
from preprocessing.normalize import normalize
from preprocessing.windowing import create_windows
from preprocessing.encode_spikes import latency_encoding


def add_symmetrical_components(X):
    """
    Zero-sequence (I0,V0): non-zero ONLY in ground faults (AG, ABG, ABCG)
    Negative-sequence (I2,V2): non-zero in asymmetric faults (AG, BC, BCG)
    These give the SNN genuinely separable features across all 6 fault classes.
    """
    Ia, Ib, Ic = X[:, 0], X[:, 1], X[:, 2]
    Va, Vb, Vc = X[:, 3], X[:, 4], X[:, 5]

    a = np.exp(1j * 2 * np.pi / 3)  # 120 degree rotation operator

    # Current symmetrical components
    I0 = np.abs((Ia + Ib + Ic) / 3)
    I1 = np.abs((Ia + a * Ib + a**2 * Ic) / 3)
    I2 = np.abs((Ia + a**2 * Ib + a * Ic) / 3)

    # Voltage symmetrical components
    V0 = np.abs((Va + Vb + Vc) / 3)
    V1 = np.abs((Va + a * Vb + a**2 * Vc) / 3)
    V2 = np.abs((Va + a**2 * Vb + a * Vc) / 3)

    sym = np.stack([I0, I1, I2, V0, V1, V2], axis=1).real
    return np.hstack([X, sym])  # shape: (N, 12)


def run_classification_pipeline(path):

    # 1. Load
    df = load_dataset(path)

    # 2. Clean
    X, y = clean_classification_data(df)
    X = X.values  # convert to numpy for symmetrical component computation

    # 3. Add symmetrical components BEFORE normalizing
    X = add_symmetrical_components(X)

    # 4. Normalize all 12 features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 5. Windowing
    X_win, y_win = create_windows(X, y, window_size=100)

    # 6. Encoding
    spikes = []
    for window in X_win:
        spikes.append(latency_encoding(window, threshold=0.3))

    spikes = np.array(spikes)

    return spikes, y_win
