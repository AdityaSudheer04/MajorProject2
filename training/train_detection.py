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