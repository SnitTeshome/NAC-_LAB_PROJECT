import sentencepiece as spm
from pathlib import Path

class SubwordTokenizer:
    """
    A wrapper class for subword-level tokenization using SentencePiece BPE.

    Provides methods for training, encoding, decoding, and retrieving subword pieces.
    """

    def __init__(self, model_file: str = None):
        """
        Initialize the tokenizer. If a model file is provided and exists, load it.

        Args:
            model_file (str, optional): Path to a pre-trained SentencePiece model file.
        """
        self.sp = spm.SentencePieceProcessor()
        if model_file and Path(model_file).exists():
            self.sp.Load(model_file)

    def train(
        self,
        input_file: str,
        model_prefix: str = "spm_model",
        vocab_size: int = 8000,
        model_type: str = "bpe"
    ):
        """
        Train a subword tokenizer on a text file and load the trained model.

        Args:
            input_file (str): Path to the training text file.
            model_prefix (str): Prefix for the SentencePiece model and vocab files.
            vocab_size (int): Desired vocabulary size.
            model_type (str): Tokenization algorithm ('bpe' or 'unigram').
        """
        spm.SentencePieceTrainer.Train(
            f"--input={input_file} --model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} --model_type={model_type}"
        )
        self.sp.Load(f"{model_prefix}.model")

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of subword IDs.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: List of subword token IDs.
        """
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of subword IDs back into text.

        Args:
            ids (list[int]): List of subword token IDs.

        Returns:
            str: The decoded string.
        """
        return self.sp.DecodeIds(ids)

    def encode_pieces(self, text: str) -> list[str]:
        """
        Encode a string into a list of subword pieces (tokens).

        Args:
            text (str): The text to encode.

        Returns:
            list[str]: List of subword token pieces.
        """
        return self.sp.EncodeAsPieces(text)


def prepare_subword_data(
    input_file: str,
    model_prefix: str = "spm_model",
    vocab_size: int = 8000,
    model_type: str = "bpe",
    force_train: bool = False
) -> tuple[list[list[int]], SubwordTokenizer]:
    """
    Wrapper function for main script: trains or loads a subword tokenizer
    and encodes all lines in a dataset into subword IDs.

    Args:
        input_file (str): Path to the training text file.
        model_prefix (str): Prefix for SentencePiece model files.
        vocab_size (int): Desired vocabulary size.
        model_type (str): Tokenization algorithm ('bpe' or 'unigram').
        force_train (bool): If True, retrain the tokenizer even if a model exists.

    Returns:
        tuple:
            - encoded_data (list[list[int]]): Encoded subword ID sequences per line.
            - tokenizer (SubwordTokenizer): The trained tokenizer object for decoding or future use.
    """
    tokenizer = SubwordTokenizer(model_file=f"{model_prefix}.model")

    if force_train or not Path(f"{model_prefix}.model").exists():
        tokenizer.train(input_file, model_prefix, vocab_size, model_type)

    encoded_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                encoded_data.append(tokenizer.encode(line))

    return encoded_data, tokenizer
