import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df


def clean_detection_data(df):
    df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"], errors='ignore')
    X = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]
    y = df["Output (S)"]
    return X, y


def clean_classification_data(df):
    X = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]
    y = df[["G", "C", "B", "A"]]
    return X, y