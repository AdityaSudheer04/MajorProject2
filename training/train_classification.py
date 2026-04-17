import torch
import torch.nn as nn
import torch.optim as optim
from models.snn_classification import SNNClassification
from preprocessing.pipeline_classification import run_classification_pipeline
from torch.utils.data import TensorDataset, DataLoader


# Load data
spikes, labels = run_classification_pipeline("data/raw/classData.csv")

X = torch.tensor(spikes, dtype=torch.float32).permute(1, 0, 2)

# Convert one-hot to class indices
y = torch.tensor(labels, dtype=torch.long)  # ← Convert to numpy first
y = torch.argmax(y, dim=1)  # ← Find which column is 1

from sklearn.model_selection import train_test_split

X_data = X.permute(1,0,2)
y_data = y

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# Model
model = SNNClassification()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
# Remove scheduler entirely

epochs = 100

for epoch in range(epochs):
    total_loss = 0
    total_acc = 0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.permute(1, 0, 2)
        optimizer.zero_grad()
        output = model(batch_X)
        final_output = output.mean(dim=0)
        loss = criterion(final_output, batch_y)
        loss.backward()
        optimizer.step()
        # REMOVE scheduler.step() from here

        preds = torch.argmax(final_output, dim=1)
        acc = (preds == batch_y).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

    # scheduler.step()  # ← Move it here (once per epoch)
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    # Test evaluation
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.permute(1, 0, 2)
            output = model(batch_X)
            final_output = output.mean(dim=0)
            loss = criterion(final_output, batch_y)
            preds = torch.argmax(final_output, dim=1)
            acc = (preds == batch_y).float().mean()
            test_loss += loss.item()
            test_acc += acc.item()

    test_acc /= len(test_loader)
    test_loss /= len(test_loader)
    
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Test Acc: {test_acc:.4f}")
    model.train()