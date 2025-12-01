import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(self, seq_len, embedding_dim, hidden1, hidden2, vocab_size):
        """
        seq_len: Number of tokens in a single input sequence (sentence length after padding).
        embedding_dim: size of token embeddings (how many different hidden features the neural network uses to describe each word.)
        hidden1/hidden2: sizes of hidden layers
        #vocab_size → Number of classes in output (the size of your vocabulary). Each neuron in the output predicts the likelihood of a particular word being the next word.
        """
        #Initialize Parent Class
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size        #vocab_size → Number of classes in output (the size of your vocabulary). Each neuron in the output predicts the likelihood of a particular word being the next word.

        

       
       # Embedding Layer: map token indices -> dense vectors
       # input size = seq_len × embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        # feedforward on flattened embeddings
        # FNNs expect 2D input [batch_size, features]:Flatten [batch_size, seq_len, embedding_dim] → [batch_size, seq_len*embedding_dim].
        # Each sentence now becomes a single vector representing all word embeddings concatenated.
        in_features = seq_len * embedding_dim


        self.layers = nn.Sequential(   # is a PyTorch container that lets you stack layers in order(pipline):Input → Layer1 → Layer2 → Layer3 → Output
            nn.Linear(in_features, hidden1),  # Each of the hidden1 neurons sees all words in the input sentence
            nn.ReLU(),   # Adds non-linearity so the network can learn complex patterns.
            nn.Linear(hidden1, hidden2), # hidden2 neurons learn from hidden1 activations.
            nn.ReLU(),   #Non-linearity again.
            nn.Linear(hidden2, vocab_size)  # logits over vocabulary(They represent the linear output of a model before it is transformed into probabilities.)
        )


    # Forward Pass
    def forward(self, x):
        # x: LongTensor [batch, seq_len]
        emb = self.embedding(x)                 # [batch, seq_len, embedding_dim]  batch_sizenumber of sentences processed together
        flat = emb.view(emb.size(0), -1)        # [batch, seq_len * embedding_dim]
        out = self.layers(flat)                    # [batch, vocab_size]
        return out