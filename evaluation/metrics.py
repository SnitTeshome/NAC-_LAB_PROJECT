"""
Evaluation metrics for next-token prediction and text generation.

Contains functions to calculate:
- Perplexity (single loss and dataset-level)
- BLEU score (sentence-level and corpus-level)
"""

import math
from typing import List
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


# --------------------------------------------------
# Perplexity
# --------------------------------------------------
def calculate_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss (float): Cross-entropy loss value.

    Returns:
        float: Perplexity score.
    """
    return math.exp(loss)


def calculate_dataset_perplexity(losses: List[float]) -> float:
    """
    Compute perplexity over a dataset using average loss.

    Args:
        losses (List[float]): List of batch or sentence losses.

    Returns:
        float: Dataset-level perplexity.
    """
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


# --------------------------------------------------
# BLEU (sentence-level)
# --------------------------------------------------
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

    score = sentence_bleu(
        [reference_tokens],
        predicted_tokens,
        smoothing_function=smoothing
    )
    return score


# --------------------------------------------------
# BLEU (corpus-level)
# --------------------------------------------------
def calculate_corpus_bleu(
    reference_sentences: List[str],
    predicted_sentences: List[str]
) -> float:
    """
    Calculate corpus-level BLEU score.

    Args:
        reference_sentences (List[str]): Ground-truth sentences.
        predicted_sentences (List[str]): Generated sentences.

    Returns:
        float: Corpus BLEU score.
    """
    references = [[ref.split()] for ref in reference_sentences]
    hypotheses = [pred.split() for pred in predicted_sentences]

    smoothing = SmoothingFunction().method1
    score = corpus_bleu(
        references,
        hypotheses,
        smoothing_function=smoothing
    )
    return score
