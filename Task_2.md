

# *Transformer Architecture — Report Summary*

## *1. Introduction*

*A Transformer is a deep learning architecture designed for processing sequential data, especially natural language. Unlike recurrent models, Transformers rely entirely on attention mechanisms, allowing them to capture long-range dependencies efficiently and process data in parallel.*

---

## *2. Input Representation*

### *2.1 Tokenization and Embeddings*

*The input text is first broken into smaller units called tokens. Each token is then mapped to a high-dimensional numerical vector known as an embedding. These embeddings encode semantic meaning and serve as the initial representation of the input.*

### *2.2 Positional Encoding*

*Since Transformers do not inherently understand word order, positional encodings are added to token embeddings. This allows the model to capture information about the position of each token in the sequence.*

---

## *3. Self-Attention Mechanism*

*The self-attention mechanism is the core component of the Transformer architecture. It allows each token to interact with all other tokens in the sequence.*

### *3.1 Queries, Keys, and Values*

*Each token embedding is linearly transformed into three vectors:*

* *Query (Q)*
* *Key (K)*
* *Value (V)*

*These vectors determine how much attention one token should pay to another.*

### *3.2 Attention Computation*

*Attention scores are computed using dot products between queries and keys. These scores are normalized using the softmax function and then applied to the value vectors. The result is a context-aware representation of each token.*

### *3.3 Masked Attention*

*In generative tasks, future tokens are masked to prevent the model from accessing information it should not yet see.*

---

## *4. Multi-Head Attention*

*Instead of performing a single attention operation, Transformers use multiple attention heads in parallel. Each head learns different relationships within the data, enabling the model to capture diverse contextual patterns simultaneously.*

---

## *5. Transformer Layer Structure*

*Each Transformer layer consists of the following components:*

* *Multi-Head Self-Attention*
* *Feed-Forward Neural Network*
* *Residual Connections*
* *Layer Normalization*

*These layers are stacked multiple times, allowing the model to build increasingly complex representations.*

---

## *6. Parallelization and Efficiency*

*Unlike recurrent models, Transformers process all tokens simultaneously. This parallel structure makes training faster and more scalable on modern hardware such as GPUs and TPUs.*

---

## *7. Output Generation*

*After passing through multiple Transformer layers, the final representations are fed into an output layer. A softmax function is applied to predict probabilities over possible next tokens or target classes.*

---

## *8. Why Transformers Are Effective*

*Transformers are effective because they:*

* *Capture global context efficiently*
* *Enable parallel computation*
* *Scale well to large datasets*
* *Generalize across domains such as text, images, and audio*

---

## *9. Overall Architecture Flow*

```text
*Input Tokens*
      ↓
*Embedding + Positional Encoding*
      ↓
*Stacked Transformer Layers*
      ↓
*Final Representation*
      ↓
*Output Prediction (Softmax)*
```

---

*This report summarizes the architecture of Transformers as presented in the talk **“Visualizing Transformers and Attention”** at TNG Big Tech Day ’24, emphasizing the intuition and structure behind attention-based models.*


### * References*
– *TNG Technology Consulting. Insights from Big Tech Day 24 talk on Transformers*
https://www.tngtech.com/en/about-us/news/insights-from-tngs-big-techday-24-talk-on-visualizing-transformers
