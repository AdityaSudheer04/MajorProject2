from preprocessing.load_data import load_dataset
import matplotlib.pyplot as plt

df = load_dataset("data/raw/detect_dataset.csv")

print(df.head(30))

plt.plot(df["Va"].values[:200])
plt.show()