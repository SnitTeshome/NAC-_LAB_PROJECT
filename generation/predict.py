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


# -------------------------------------------
# Subword-level sentence generation function
# -------------------------------------------
def generate_sentence_subword(
    model,
    tokenizer,
    seed_text,
    max_gen_len=20,
    device="cpu"
):
    """
    Generate a sentence using subword prediction iteratively.

    Args:
        model: Trained PyTorch model
        tokenizer: Subword tokenizer (SentencePiece)
        seed_text: String of initial text
        max_gen_len: Maximum number of subwords to generate
        device: "cpu" or "cuda"

    Returns:
        str: Generated sentence
    """
    # Encode seed text into subword IDs
    input_ids = tokenizer.encode(seed_text)
    generated_ids = input_ids.copy()

    for _ in range(max_gen_len):
        pred_id, pred_piece, pred_text = predict_next_subword(
            model, tokenizer, generated_ids, device=device
        )

        # Stop if end-of-sentence token appears
        if pred_text == "<EOS>":
            break

        generated_ids.append(pred_id)

    # Decode all generated IDs into text
    generated_sentence = tokenizer.decode(generated_ids)
    return generated_sentence
