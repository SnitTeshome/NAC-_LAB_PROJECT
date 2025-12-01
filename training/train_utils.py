import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim



def train_model(model, train_inputs, train_outputs, val_inputs, val_outputs,
                batch_size=64, epochs=5, lr=1e-3, device=None):
    # Set device:
    device = "cpu"
    # convert to tensors:✔ Converts NumPy arrays → PyTorch tensors
    #✔ dtype = long because embeddings require integer indices
    X_train = torch.tensor(train_inputs, dtype=torch.long)   # [num_sentences, seq_len-1]
    Y_train = torch.tensor(train_outputs, dtype=torch.long)  # but shifted by 1 step (next-word targets)
    X_val   = torch.tensor(val_inputs, dtype=torch.long) 
    Y_val   = torch.tensor(val_outputs, dtype=torch.long)

     #TensorDataset always pairs corresponding rows from the tensors you pass in:
     # (X_train[i], Y_train[i])  :y_target = Y_train[i][0] for next-word prediction
     # It will return :(sentence_input, shifted_sentence_output).
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val, Y_val)

    # The DataLoader takes your TensorDataset:
           # ✔ Breaks the dataset into mini-batches:batch_size=batch_size(Mini-batches give the best trade-off.)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
   
    criterion = nn.CrossEntropyLoss()
     #takes logits [batch, vocab_size]
    # takes targets [batch] (word index)
    # compares prediction word vs correct next word
    optimizer = optim.Adam(model.parameters(), lr=lr)   #Adam update:  w ← w − lr * adaptive_scaled_gradient


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:   # Automatically feeds the batches into the training loop:xb → a batch of inputs  and yb → corresponding outputs
            xb = xb.to(device)                # LongTensor [batch, seq_len]
            yb = yb.to(device)

            # Select the target token you want to predict.
            # Notebook used first "next-word" => yb[:,0]
            y_target = yb[:, 0].long()        # [batch]

            # Forward Pass
            logits = model(xb)                # [batch, vocab_size], float


            #Compute Loss + Backpropagation
            loss = criterion(logits, y_target)

            optimizer.zero_grad()   #Because PyTorch accumulates gradients by default.
            loss.backward()    #uses to compute gradients:
                 #The computational graph (autograd graph):
                 # Every operation stores:
                 # what function was applied ,what inputs and outputs were gradients formulas needed references to parents and children.
            optimizer.step()   #actually applies those changes to the parameters.

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Validation
        model.eval()  # Switch model to evaluation mode.
        val_loss = 0.0
        with torch.no_grad():    # Disable gradient computation.
            for xb, yb in val_loader:    # terate over batches from the validation DataLoader.
                xb = xb.to(device)
                yb = yb.to(device)
                y_target = yb[:, 0].long()  # Select the **first token** in the output sequence as the target.
                logits = model(xb)  # Forward pass: compute model predictions (no backprop).
                loss = criterion(logits, y_target)
                val_loss += loss.item() * xb.size(0)   
        #  Compute average validation loss across the entire dataset
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")