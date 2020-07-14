# Fake News Detection Through Transformer Model

Transformer was originally proposed for the task of machine translation in NIPS-2017 paper "Attention Is All You Need".

Here, I have implemented the Transformer Model for fake news detection


## Transformer Overview
- Encoder-decoder architecture
- Final cost function is standard Cross Entropy error on top of a softmax classifier

## Basic Building Block - Dot Product Attention
**Inputs:** </br>
- A query *q*
- Set of key-value (*k-v*) pairs to an output
- Query, keys, values and output are all vectors

**Output:** </br>
- Weighted sum of values
- Weight of each value is computed using an inner product of query and corresponding key</br>
- Query and keys have dimensionality d<sub>k</sub>
- Values have dimensionality d<sub>v</sub>

### Dot-Product Attention - Matrix Notation
- When we have multiple queries *q*, we stack them in a matrix *Q*.

### Scaled Dot-Product Attention
-> some values inside the softmax get large</br>
-> the softmax gets very peaked</br>
-> hence its gradient gets smaller

**Solution:** Scale by length of query/key vectors.
## Self-Attention and Multi-head Attention
- Input word vectors could be the queries, keys and values
- The word vectors themselves select each other
- Word vector stacks - Q = K = V
- **Problem:** Only one way for words to interact with each other
- **Solution:** Multi-head attention
- First map Q,K,V into *h* many lower dimensional spaces via **W** matrices
- Then apply attention, then concatenate outputs and pipe through linear layer

## Encoder Input
- Sentences are encoded using byte-pair encodings
- For the model to make use of the order of the sequence, positional encoding is used

## Complete Encoder
- Encoder is composed of *N*=1 identical layers
- Each layer has 2 sub-layers
(1) Multi-head attention
(2) 2-layer feed-forward network
- Each sublayer also has
(1) Residual (short-circuit)
(2) Layer normalization
- i.e. output of each sublayer is **LayerNorm(*x*+Sublayer(x))**
- In a self-attention layer, all the **keys**, **values** and **queries** come from the same place - the output of previous layer in the encoder.
- Each position in the encoder can attend to all positions in the previous layer of the encoder

## Complete Decoder
- Similar to encoder, decoder is also composed of *N*=6 identical layers
- Each layer of decoder has 3 sub-layers
(1) Maksed multi-head attention over previous decoder outputs
(2) Multi-head attention over output of encoder
(3) 2-layer feed-forward network
- Each sublayer also has
(1) Residual (short-circuit)
(2) Layer normalization
- i.e. output of each sublayer is **LayerNorm(*x*+Sublayer(x))**
- In encoder-decoder attention, **queries** come from previous decoder layer, **keys** and **values** come from output of encoder 
- This allows every position in the decoder to attend over all positions in the input sequence
- Self-attention in the decoder allow each position in decoder to attend to all positions in the decoder *up to and including that position*.
- We need to prevent leftward information flow in the decoder to preserve the auto-regressive property
The dataset used in this study got many short-labeled statements from API of PolitiFact.com. 