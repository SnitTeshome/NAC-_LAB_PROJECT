# data_utils.py
"""
Data Preprocessing Utilities for Next-Word Prediction (Penn Treebank)
---------------------------------------------------------------------
This module handles:
1. Loading and tokenizing text files
2. Building a vocabulary (word ↔ index mappings)
3. Preparing padded input-output pairs for training
"""

import numpy as np
from pathlib import Path
from nltk.tokenize import word_tokenize


def load_and_tokenize(file_path):
    """
    Reads a text file and splits it into tokenized sentences.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: A list of tokenized sentences (each sentence = list of tokens).
        A Python list where:
        - each item = one sentence
        - each sentence = a list of words
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = word_tokenize(line.strip())
            if tokens:
                sentences.append(tokens)
    return sentences


def build_vocab(sentences):
    """
    Builds a mapping between words and numerical indices.

    Args:
        sentences (list): List of tokenized sentences.

    Returns:
        tuple: (word_to_index, index_to_word)
    """
    all_tokens = [token for sent in sentences for token in sent]  # loop over each sentence (list)  then  # loop over each word inside the sentence   and append 
    vocab = set(all_tokens)     # unique words in the dataset
    word_to_index = {"<PAD>": 0, "<UNK>": 1}
               # <PAD> = 0 → used to pad sentences to same length.

              # <UNK> = 1 → used for unknown words not in the vocabulary.

    for idx, word in enumerate(sorted(vocab), start=2):
        word_to_index[word] = idx

    index_to_word = {idx: word for word, idx in word_to_index.items()} # word_to_index.items():[("<PAD>", 0), ("<UNK>", 1), ("a", 2), ("cat", 3), ("dog", 4)]

    return word_to_index, index_to_word


def prepare_data(sentences, word_to_index, max_len=None):
    """
    Convert tokenized sentences into input-output sequences for next-word prediction.

    Parameters:
    - sentences: list of tokenized sentences
    - word_to_index: dictionary mapping words to integer indices
    - max_len: fixed maximum sequence length (padding/truncating)

    Returns:
    - inputs: numpy array of shape (num_sentences, max_len-1)
    - outputs: numpy array of shape (num_sentences, max_len-1)
    """
     # Compute longest sentence length
    if max_len is None:
        max_len = max(len(s) for s in sentences)

    inputs, outputs = [], []

    for sentence in sentences:

        # Convert tokens → integers, <UNK> = 1
        indices = [word_to_index.get(token, 1) for token in sentence]   #.get(key, default):default value if key not found==>1

        # Apply padding or truncation
        if len(indices) > max_len:
            indices = indices[:max_len]    #truncate long sentence
        else:
            indices += [0] * (max_len - len(indices))  # pad short sentence with <PAD> = 0    

        # Create shifted input/output pairs
        inputs.append(indices[:-1])  #inputs → all words except the last in the sentence.
        outputs.append(indices[1:])  #outputs → all words except the first in the sentence.

    return np.array(inputs), np.array(outputs), max_len

#Input array shape: (num_sentences, max_len-1)

# Output array shape: (num_sentences, max_len-1)

# Each sentence → one training example (a sequence of tokens)

# Each token in the sentence → one prediction target for next-word prediction

