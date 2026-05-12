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

epochs = 4

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

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {avg_acc:.4f} | Test Acc: {test_acc:.4f}")

FAULT_NAMES = {
    '0-0-0-0': 'No Fault',
    '0-1-1-0': 'BC Fault (LL)',
    '0-1-1-1': 'ABC Fault (LLL)',
    '1-0-0-1': 'AG Fault (LG)',
    '1-0-1-1': 'ABG Fault (LLG)',
    '1-1-1-1': 'ABCG Fault (LLLG)',
}

idx_to_name = {i: FAULT_NAMES.get(cls, cls) for i, cls in enumerate(le.classes_)}

print("\n" + "="*60)
print("       POST-TRAINING SINGLE SAMPLE INFERENCE DEMO")
print("="*60)

model.eval()
with torch.no_grad():

    # Pick a random sample from the test set
    rand_idx = torch.randint(0, len(X_test), (1,)).item()
    sample_X = X_test[rand_idx]           # shape: (T, 12)
    true_label = y_test[rand_idx].item()  # integer class index

    # Prepare input: add batch dimension → (T, 1, 12)
    input_tensor = sample_X.unsqueeze(1).permute(1, 0, 2)  # (1, T, 12)
    input_tensor = input_tensor.permute(1, 0, 2)            # (T, 1, 12)

    # Forward pass
    spk_out, mem_out = model(input_tensor)

    # Spike count vector across all 6 classes
    spike_counts = spk_out.squeeze(0)         # shape: (6,)
    predicted_class = torch.argmax(spike_counts).item()
    confidence = torch.softmax(spike_counts, dim=0)[predicted_class].item()

    # ── Print raw input (first and last timestep for brevity) ──
    print(f"\n INPUT SAMPLE (index {rand_idx} from test set)")
    print(f"   Feature order: Ia  Ib  Ic  Va  Vb  Vc  I0  I1  I2  V0  V1  V2")
    print(f"   Timestep   0 : {sample_X[0].numpy().round(3)}")
    print(f"   Timestep  49 : {sample_X[49].numpy().round(3)}")
    print(f"   Timestep  99 : {sample_X[99].numpy().round(3)}")
    print(f"   (showing t=0, t=49, t=99 out of {sample_X.shape[0]} timesteps)")

    # ── Print spike counts per class ──
    print(f"\n SPIKE COUNTS PER CLASS (out of T=100 timesteps):")
    for i, count in enumerate(spike_counts.numpy()):
        marker = " ← predicted" if i == predicted_class else ""
        print(f"   Class {i} [{idx_to_name[i]:20s}]: {int(count):3d} spikes{marker}")

    # ── Print result ──
    print(f"\n TRUE LABEL     : Class {true_label} → {idx_to_name[true_label]}")
    print(f" PREDICTED      : Class {predicted_class} → {idx_to_name[predicted_class]}")
    print(f" CONFIDENCE     : {confidence*100:.1f}%")

    if predicted_class == true_label:
        print(f" RESULT         : CORRECT")
    else:
        print(f" RESULT         : WRONG")
        print(f"   (Model predicted {idx_to_name[predicted_class]} "
              f"but true fault was {idx_to_name[true_label]})")

print("="*60)
