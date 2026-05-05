import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.snn_classification import SNNClassification
from preprocessing.pipeline_classification import run_classification_pipeline
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# Load data
spikes, labels = run_classification_pipeline("data/raw/classData.csv")

from sklearn.preprocessing import LabelEncoder

X = torch.tensor(spikes, dtype=torch.float32).permute(1, 0, 2)  # (time, batch, 12 features)

# ✅ Correct label encoding — treat each unique G/C/B/A combination as its own class
# Your data has 6 fault types: No fault, AG, ABG, BC, BCG, ABCG
# argmax was WRONG — it only found the first 1 in each row, collapsing 6 classes into 2
label_array = np.array(labels)  # shape: (N, 4)
label_strings = ['-'.join(map(str, row)) for row in label_array]
le = LabelEncoder()
y_encoded = le.fit_transform(label_strings)
num_classes = len(le.classes_)

print(f"=== Fault classes found: {num_classes} ===")
for i, cls in enumerate(le.classes_):
    print(f"  Class {i}: G-C-B-A = {cls} → count: {np.sum(y_encoded == i)}")

y = torch.tensor(y_encoded, dtype=torch.long)

X_data = X.permute(1, 0, 2)
y_data = y

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data  # ✅ stratify keeps balance
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# ✅ Compute class weights from full dataset with correct num_classes
all_classes = np.arange(num_classes)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=all_classes,
    y=y_data.numpy()
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"\nClass weights: {class_weights_tensor}")

# Model — output size must match num_classes (6), not hardcoded 4
model = SNNClassification(num_classes=num_classes)

# ✅ Pass class weights into CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.permute(1, 0, 2)
        optimizer.zero_grad()

        # ✅ spk_out = summed spike counts (batch, 4)
        # ✅ mem_out = final membrane potential at last timestep (batch, 4)
        spk_out, mem_out = model(batch_X)

        # ✅ Correct weighted loss: spike COUNT loss + membrane potential loss
        loss_spk = criterion(spk_out, batch_y)    # spike count → class
        loss_mem = criterion(mem_out, batch_y)    # membrane potential → class

        loss = 0.7 * loss_spk + 0.3 * loss_mem

        loss.backward()
        optimizer.step()

        # ✅ Predictions from spike counts (the primary signal)
        preds = torch.argmax(spk_out, dim=1)
        acc = (preds == batch_y).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    # Evaluation
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.permute(1, 0, 2)
            spk_out, mem_out = model(batch_X)

            loss_spk = criterion(spk_out, batch_y)
            loss_mem = criterion(mem_out, batch_y)
            loss = 0.7 * loss_spk + 0.3 * loss_mem

            preds = torch.argmax(spk_out, dim=1)
            acc = (preds == batch_y).float().mean()
            test_loss += loss.item()
            test_acc += acc.item()

    test_acc /= len(test_loader)
    test_loss /= len(test_loader)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | Test Acc: {test_acc:.4f}")
