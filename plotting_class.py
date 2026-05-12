import re
import matplotlib.pyplot as plt

# Read file
with open("snn_cla.txt", "r") as f:
    log_data = f.readlines()

epochs = []
train_acc = []
test_acc = []

# Extract values
for line in log_data:
    match = re.search(
        r"Epoch (\d+)/\d+ \| Train Loss: .*? \| Train Acc: ([\d.]+) \| Test Acc: ([\d.]+)",
        line
    )

    if match:
        epochs.append(int(match.group(1)))
        train_acc.append(float(match.group(2)))
        test_acc.append(float(match.group(3)))

# Plot
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_acc, label='Training Accuracy', linewidth=2)
plt.plot(epochs, test_acc, label='Testing Accuracy', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.legend()
plt.grid(True)

plt.show()