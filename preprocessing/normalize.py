from sklearn.preprocessing import MinMaxScaler

def normalize(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)