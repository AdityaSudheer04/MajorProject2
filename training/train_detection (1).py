import torch
import torch.nn as nn
import torch.optim as optim
from models.snn_detection import SNNDetection
from preprocessing.pipeline import run_pipeline
from torch.utils.data import TensorDataset, DataLoader



# Load data
spikes, labels = run_pipeline("data/raw/detect_dataset.csv")

X = torch.tensor(spikes, dtype=torch.float32).permute(1, 0, 2)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

from sklearn.model_selection import train_test_split

X_data = X.permute(1,0,2)
y_data = y

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
# Model
model = SNNDetection()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 50

for epoch in range(epochs):
    total_loss = 0
    total_acc = 0
    num_batches = 0  # ← Add this

    for batch_X, batch_y in train_loader:

        # reshape back to (time, batch, features)
        batch_X = batch_X.permute(1, 0, 2)

        optimizer.zero_grad()

        output = model(batch_X)
        final_output = output.mean(dim=0)

        loss = criterion(final_output, batch_y)

        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(final_output) > 0.5
        acc = (preds.float() == batch_y).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1  # ← Count batches

    avg_loss = total_loss / num_batches        # ← Average
    avg_acc = total_acc / num_batches          # ← Average
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    # Add after each epoch
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.permute(1, 0, 2)
            output = model(batch_X)
            final_output = output.mean(dim=0)
            loss = criterion(final_output, batch_y)
            preds = torch.sigmoid(final_output) > 0.5
            acc = (preds.float() == batch_y).float().mean()
            test_loss += loss.item()
            test_acc += acc.item()

    test_acc /= len(test_loader)
    test_loss /= len(test_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Test Acc: {test_acc:.4f}")
    model.train()
# ─────────────────────────────────────────────────────────────────────────────
# POST-TRAINING: Single sample inference demo
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("       POST-TRAINING SINGLE SAMPLE INFERENCE DEMO")
print("="*60)

model.eval()
with torch.no_grad():

    # Pick a random sample from the test set
    rand_idx = torch.randint(0, len(X_test), (1,)).item()
    sample_X = X_test[rand_idx]            # shape: (T, 12)
    true_label = y_test[rand_idx].item()   # 0 or 1

    # Prepare input: (T, 1, 12)
    input_tensor = sample_X.unsqueeze(0).permute(1, 0, 2)  # (T, 1, 12)

    # Forward pass
    output = model(input_tensor)           # (T, 1, 1)
    final_output = output.mean(dim=0)      # (1, 1)
    probability = torch.sigmoid(final_output).item()
    predicted = 1 if probability > 0.5 else 0

    # ── Print raw input ──
    print(f"\n📥 INPUT SAMPLE (index {rand_idx} from test set)")
    print(f"   Feature order: Ia  Ib  Ic  Va  Vb  Vc  I0  I1  I2  V0  V1  V2")
    print(f"   Timestep   0 : {sample_X[0].numpy().round(3)}")
    print(f"   Timestep  49 : {sample_X[49].numpy().round(3)}")
    print(f"   Timestep  99 : {sample_X[99].numpy().round(3)}")
    print(f"   (showing t=0, t=49, t=99 out of {sample_X.shape[0]} timesteps)")

    # ── Print membrane potential summary ──
    mem_vals = output.squeeze().numpy()
    print(f"\n⚡ MEAN MEMBRANE POTENTIAL : {final_output.item():.4f}")
    print(f"   Min across timesteps   : {mem_vals.min():.4f}")
    print(f"   Max across timesteps   : {mem_vals.max():.4f}")
    print(f"   Sigmoid probability    : {probability:.4f}")

    # ── Print result ──
    true_str = "FAULT" if true_label == 1 else "NO FAULT"
    pred_str = "FAULT" if predicted == 1 else "NO FAULT"

    print(f"\n🎯 TRUE LABEL  : {true_label} → {true_str}")
    print(f"🤖 PREDICTED   : {predicted} → {pred_str}  (p={probability:.4f})")

    if predicted == true_label:
        print(f"✅ RESULT      : CORRECT")
    else:
        print(f"❌ RESULT      : WRONG")
        print(f"   (Model predicted {pred_str} but true label was {true_str})")

print("="*60)
