import torch

def predict_next_subword(model, tokenizer, input_ids, device="cpu"):
    """
    Predict the next subword token using a trained model and a subword tokenizer.

    Args:
        model: Trained PyTorch model for next-token prediction.
        tokenizer: SubwordTokenizer instance (SentencePiece).
        input_ids (list[int]): List of already encoded subword token IDs.
        device (str): "cpu" or "cuda".

    Returns:
        tuple: (predicted_id, predicted_piece, predicted_text)
            predicted_id (int): Subword token ID
            predicted_piece (str): The raw SentencePiece token (e.g., "▁play")
            predicted_text (str): Decoded text for that single token
    """
    
    model.eval()
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        # logits shape: (1, seq_len, vocab_size)
        predicted_id = torch.argmax(logits[0, -1]).item()

    # Convert predicted ID → subword token
    predicted_piece = tokenizer.sp.IdToPiece(predicted_id)
    predicted_text = tokenizer.decode([predicted_id])

    return predicted_id, predicted_piece, predicted_text
