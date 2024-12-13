

This project implements a seq2seq neural network with bahdanau attention for text generation. The architecture is composed of the following components:  
The attention mechanism dynamically focuses on relevant parts of the input sequence during decoding and operates within an Encoder-Decoder architecture and consists of three main components:

1. **Encoder**: Processes the input sequence and generates hidden states.
2. **Attention Mechanism**: Computes a context vector dynamically at each decoding step.
3. **Decoder**: Uses the context vector to generate the output sequence.


1. Encoder 
Embedding Layer: Converts input tokens into dense vectors of fixed size.
GRU Layer: A gated recurrent unit processes the embedded input sequence and encodes the information into hidden states. The GRU is configured with:
Return of both sequences and the final hidden state.
Initialization with the Glorot Uniform initializer.
Hidden State Initialization: The initial hidden state is a tensor of zeros with shape (batch_size, enc_units).
Output:

Sequence of hidden states with shape (batch_size, sequence_length, enc_units).
Final hidden state with shape (batch_size, enc_units).  
\[
h_i = \text{BiRNN}(X_i), \quad h_i = [\overrightarrow{h_i}; \overleftarrow{h_i}]
\]
where \(h_i\) is the concatenation of forward (\(\overrightarrow{h_i}\)) and backward (\(\overleftarrow{h_i}\)) RNN states.

  
2. Attention mechanism
The architecture employs Bahdanau Attention to enhance the decoder's ability to focus on relevant parts of the input sequence at each decoding step.

Dense layers: Two fully connected layers compute a score for each encoder output.
Alignement score calculation: Applies tanh to combine the decoder's hidden state with encoder outputs.  
The alignment score \(e_{t,i}\) measures the compatibility between the decoder state \(s_t\) and encoder hidden state \(h_i\):  
\[
e_{t,i} = v_a^\top \tanh(W_a s_t + U_a h_i)
\]
where:
- \(W_a\) and \(U_a\) are learned weight matrices.
- \(v_a\) is a learned vector.
Softmax Attention Weights: Normalized weights indicate the importance of each time step in the input sequence.  
\[
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^T \exp(e_{t,j})}
\]
- context vector \(c_t\) : a weighted sum of the encoder hidden states:
\[
c_t = \sum_{i=1}^T \alpha_{t,i} h_i
\]
Output:

Context vector of shape (batch_size, hidden_size).
Attention weights of shape (batch_size, sequence_length, 1).  
  
3. Decoder
Embedding Layer: Maps target tokens into dense vectors.
Attention Layer: Integrates the context vector from the encoder with the decoder's hidden state.
GRU Layer: Processes the concatenated context vector and target embeddings to generate the decoder's output.
Dense Layer: Maps the GRU output to the target vocabulary size to predict the next token.
Output:

Predicted tokens for the target sequence.
Final hidden state to be fed into the next time step.
