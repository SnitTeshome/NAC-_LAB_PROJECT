import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def test_model(model, test_inputs, test_outputs, batch_size=64, device="cpu"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    X_test = torch.tensor(test_inputs, dtype=torch.long)
    Y_test = torch.tensor(test_outputs, dtype=torch.long)
    test_ds = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_target = yb[:, 0].long()  # predict the first next word
            logits = model(xb)
            loss = criterion(logits, y_target)
            test_loss += loss.item() * xb.size(0)
    
    avg_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}")
