import numpy as np

def create_windows(X, y, window_size=20):

    windows = []
    labels = []

    for i in range(len(X) - window_size):
        windows.append(X[i:i+window_size])
        labels.append(y.iloc[i+window_size]) 

    return np.array(windows), np.array(labels)