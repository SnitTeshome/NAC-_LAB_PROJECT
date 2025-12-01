import torch

def predict_next_word(model, input_sequence, word_to_index, index_to_word, device="cpu"):
    """
    Predict the next word for a given input sequence using the trained model.
    
    Args:
        model: trained PyTorch model
        input_sequence: list of word indices (a sequence)
        word_to_index: dict mapping words → indices
        index_to_word: dict mapping indices → words
        device: "cpu" or "cuda"

    Returns:
        str: predicted next word
    """
    model.eval()
    input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_index = torch.argmax(logits, dim=1).item()
    
    return index_to_word.get(predicted_index, "<UNK>")
