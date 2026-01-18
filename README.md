
# *NAC-LAB Project*

*Next-Word Prediction & Subword Generation using Feedforward Neural Networks (FNN)*


---

## *Assigned Tasks*

*Assigned tasks included:*

1. *Developing a conceptual understanding of neural networks and the backpropagation algorithm.*
2. *Implementing a simple prototype feedforward neural network with:*

   * *1 input layer*
   * *2 hidden layers*
   * *1 output layer*
3. *Dataset: Using the Penn Treebank dataset from Kaggle, with the data already pre-split for training, validation, and testing. The dataset is stored in the `ptbdataset` folder:*

   * *`ptb.train.txt` → Training set*
   * *`ptb.valid.txt` → Validation set*
   * *`ptb.test.txt` → Test set*

---

## *Project Overview*

*This project implements a text generation pipeline using a feedforward neural network (FNN) trained on the Penn Treebank dataset. The model predicts the next word (or subword) in a sequence and can generate coherent sentences.*

*Key Features:*

* *Preprocessing: Tokenization, vocabulary building, and sequence preparation.*
* *Model: Feedforward neural network with embeddings and two hidden layers.*
* *Training: Batched training with validation and loss tracking.*
* *Evaluation: Test loss, perplexity, and BLEU score calculation.*
* *Inference: Generate next words or full sentences using the trained model.*
* *Visualization: Plot training and validation loss curves.*

---

## *Project Structure*

```
NAC-LAB_PROJECT/
│
├── data/
│   └── ptdataset/
│       ├── __init__.py
│       ├── ptb.train.txt
│       ├── ptb.valid.txt
│       └── ptb.test.txt
│
├── notebooks/
│   ├── Main.ipynb
│   ├── model.pth
│   └── evaluation/
│
├── preprocessing/
│   ├── __init__.py
│   └── data_utils.py
│
├── models/
│   ├── __init__.py
│   └── model.py
│
├── training/
│   ├── __init__.py
│   └── train_utils.py
│
├── evaluation/
│   ├── __init__.py
│   ├── test_utils.py
│   ├── metrics.py
│   └── plots.py
│
├── generation/
│   ├── __init__.py
│   └── predict.py    
├── requirements.txt
└── README.md
```



## *Installation & Setup*

```bash
# Clone the repository
git clone <your-repo-url>
cd NAC-LAB_PROJECT

# (Optional) Create virtual environment
python -m venv .myenv
source .myenv/bin/activate   # Linux/Mac
.myenv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## *How to Run*

### *1. Preprocessing & Data Preparation*

```python
from preprocessing.data_utils import load_and_tokenize, build_vocab, prepare_data

train_sentences = load_and_tokenize("data/ptdataset/ptb.train.txt")
val_sentences   = load_and_tokenize("data/ptdataset/ptb.valid.txt")
test_sentences  = load_and_tokenize("data/ptdataset/ptb.test.txt")

word_to_index, index_to_word = build_vocab(train_sentences)
train_inputs, train_outputs, max_seq_len = prepare_data(train_sentences, word_to_index)
val_inputs, val_outputs, _ = prepare_data(val_sentences, word_to_index, max_len=max_seq_len)
```

### *2. Initialize & Train the Model*

```python
from models.model import FeedforwardNN
from training.train_utils import train_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FeedforwardNN(
    seq_len=max_seq_len-1,
    embedding_dim=50,
    hidden1=128,
    hidden2=64,
    vocab_size=len(word_to_index)
).to(device)

train_losses, val_losses = train_model(
    model,
    train_inputs, train_outputs,
    val_inputs, val_outputs,
    batch_size=64,
    epochs=5,
    lr=1e-3,
    device=device
)
```

### *3. Plot Training & Validation Loss*

```python
from evaluation.plots import plot_loss

plot_loss(train_losses, val_losses, out_path="evaluation/plots/loss_curve.png")
```

### *4. Evaluate on Test Set*

```python
from evaluation.test_utils import test_model
from evaluation.metrics import calculate_perplexity

test_loss = test_model(model, test_inputs, test_outputs, device=device)
test_perplexity = calculate_perplexity(test_loss)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Perplexity: {test_perplexity:.4f}")
```

### *5. Generate Text*

```python
from generation.predict import generate_sentence

seed_seq = train_inputs[0]
seed_words = [index_to_word[idx] for idx in seed_seq[:3]]

generated_text = generate_sentence(
    model=model,
    seed_text=seed_words,
    max_gen_len=20,
    device=device
)

print("Generated sentence:", generated_text)
```

### *6. Compute BLEU Score*

```python
from evaluation.metrics import calculate_bleu

reference_sentence = " ".join(test_sentences[0])
bleu_score = calculate_bleu(reference_sentence, generated_text)

print(f"BLEU score (first test sentence): {bleu_score:.4f}")
```

### *7. Save the Trained Model*

```python
torch.save(model.state_dict(), "model.pth")
```

---


