import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_inputs, train_outputs, val_inputs, val_outputs,
                batch_size=64, epochs=5, lr=1e-3, device=None):
    """
    Train a next-word prediction model using PyTorch.

    Returns:
        train_loss_list (list[float]): Training loss per epoch
        val_loss_list (list[float]): Validation loss per epoch
    """

    # Select device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Convert NumPy arrays → tensors
    X_train = torch.tensor(train_inputs, dtype=torch.long)
    Y_train = torch.tensor(train_outputs, dtype=torch.long)
    X_val   = torch.tensor(val_inputs, dtype=torch.long)
    Y_val   = torch.tensor(val_outputs, dtype=torch.long)

    # Dataset + DataLoader
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lists to store loss per epoch
    train_loss_list = []
    val_loss_list = []

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            y_target = yb[:, 0].long()  # first token as next-word target
            logits = model(xb)
            loss = criterion(logits, y_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                y_target = yb[:, 0].long()
                logits = model(xb)
                loss = criterion(logits, y_target)
                val_loss += loss.item() * xb.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_loss_list.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Return losses for plotting
    return train_loss_list, val_loss_list
