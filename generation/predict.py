import torch


# --------------------------------------------------
# Predict the next WORD (word-level, not subword)
# --------------------------------------------------
def predict_next_word(model, input_ids, device="cpu"):
    """
    Predict the next word ID given a sequence of word IDs.

    Args:
        model (torch.nn.Module): Trained FeedforwardNN
        input_ids (list[int]): Sequence of word indices (length = seq_len)
        device (str): "cpu" or "cuda"

    Returns:
        int: Predicted word index
    """
    model.eval()

    # Shape: (1, seq_len)
    x = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)               # (1, vocab_size)
        next_word_id = torch.argmax(logits, dim=1).item()

    return next_word_id


# --------------------------------------------------
# Generate a sentence word-by-word
# --------------------------------------------------
def generate_sentence(
    model,
    seed_text,
    word_to_index,
    index_to_word,
    max_seq_len,
    max_gen_len=20,
    device="cpu"
):
    """
    Generate a sentence by predicting one word at a time.

    Args:
        model (torch.nn.Module): Trained FeedforwardNN
        seed_text (list[str]): Initial words (e.g. ["the", "market", "is"])
        word_to_index (dict): word → index mapping
        index_to_word (dict): index → word mapping
        max_seq_len (int): Input sequence length used during training
        max_gen_len (int): Maximum number of words to generate
        device (str): "cpu" or "cuda"

    Returns:
        str: Generated sentence
    """

    # Convert seed words → indices
    input_ids = [word_to_index[w] for w in seed_text if w in word_to_index]

    # Pad if seed is shorter than seq_len
    if len(input_ids) < max_seq_len:
        input_ids = [0] * (max_seq_len - len(input_ids)) + input_ids

    generated_ids = input_ids.copy()

    for _ in range(max_gen_len):
        # Use last max_seq_len words
        current_input = generated_ids[-max_seq_len:]

        next_id = predict_next_word(
            model,
            current_input,
            device=device
        )

        # Stop if padding or end token (optional)
        if next_id == 0:
            break

        generated_ids.append(next_id)

    # Convert indices → words (ignore padding)
    generated_words = [
        index_to_word[idx]
        for idx in generated_ids
        if idx != 0
    ]

    return " ".join(generated_words)
