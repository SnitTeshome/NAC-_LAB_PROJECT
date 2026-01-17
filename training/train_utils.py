import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_inputs, train_outputs, val_inputs, val_outputs,
                batch_size=64, epochs=5, lr=1e-3, device=None):
    """
    Train a next-word prediction model using PyTorch.

    This function:
    - Converts NumPy arrays into PyTorch tensors
    - Creates DataLoader objects for batching
    - Runs the training loop
    - Computes training and validation loss each epoch

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.

    train_inputs : np.ndarray
        Input sequences for training, shape (N, seq_len-1).

    train_outputs : np.ndarray
        Target sequences for training, shape (N, seq_len-1).
        Only the first target token (next-word target) is used.

    val_inputs : np.ndarray
        Input sequences for validation.

    val_outputs : np.ndarray
        Target sequences for validation.

    batch_size : int, optional (default=64)
        How many samples per batch.

    epochs : int, optional (default=5)
        Number of passes through the full training dataset.

    lr : float, optional (default=1e-3)
        Learning rate for the Adam optimizer.

    device : str or None, optional
        "cpu" or "cuda". If None, automatically selects GPU if available.

    Returns
    -------
    None
        The model is trained in-place. All results are printed to console.

    Notes
    -----
    - CrossEntropyLoss expects logits of shape (batch, vocab_size)
      and target indices of shape (batch,).
    - Only the first token of the output sequence is used as the
      next-word prediction target.
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

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # Next-word target = first shifted token
            y_target = yb[:, 0].long()

            logits = model(xb)  # (batch, vocab_size)
            loss = criterion(logits, y_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

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

        val_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")
