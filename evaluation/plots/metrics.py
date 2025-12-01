"""
Evaluation metrics for next-word prediction and text generation.

Contains functions to calculate:
- Perplexity
- BLEU score
"""

import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss (float): Cross-entropy loss value.

    Returns:
        float: Perplexity score.
    """
    return math.exp(loss)


def calculate_bleu(reference_sentence: str, predicted_sentence: str) -> float:
    """
    Calculate BLEU score between a reference sentence and a predicted sentence.

    Args:
        reference_sentence (str): Ground-truth sentence.
        predicted_sentence (str): Model-generated sentence.

    Returns:
        float: BLEU score (0.0 - 1.0).
    """
    reference_tokens = reference_sentence.split()
    predicted_tokens = predicted_sentence.split()
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([reference_tokens], predicted_tokens, smoothing_function=smoothing)
    return score
