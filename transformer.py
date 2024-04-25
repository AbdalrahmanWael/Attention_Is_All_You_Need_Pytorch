import math, copy, torch
from torch import nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, src_vocab_length, trg_vocab_length, max_length, d_model, n_heads,
                  n_layers, bias=True, dropout_p=0.1):
        super().__init__()

        # Embed Token Ids into embedding vectors
        self.src_embedding = Embedding(src_vocab_length, d_model)
        self.trg_embedding = Embedding(trg_vocab_length, d_model)

        # Adding Positional Information Using Positional Encoding
        self.src_positional_encoding = PositionalEncoding(d_model, max_length, dropout_p)
        self.trg_positional_encoding = PositionalEncoding(d_model, max_length, dropout_p)

        # Multi Head Attention
        mha = MultiHeadAttention(d_model, n_heads, bias, dropout_p)

        # Position Wise Feed Forward Network
        feed_forward = PositionwiseFeedForward(d_model, expansion_multiplier=4, bias=True, dropout_p=dropout_p)

        # Encoder Layer, Decoder Layer
        encoder_layer = TransformerEncoderLayer(d_model, mha, feed_forward, bias, dropout_p)
        decoder_layer = TransformerDecoderLayer(d_model, mha, feed_forward, bias, dropout_p)

        # Encoder, Decoder
        self.encoder = TransformerEncoder(encoder_layer, n_layers, bias) 
        self.decoder = TransformerDecoder(decoder_layer, n_layers, bias) 

        # Convert output of Decoder into log-proabilities of the trg_vocab_length)
        self.output_linear = nn.Linear(d_model, trg_vocab_length)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Apply xavier initialization
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_tokens, trg_tokens, src_mask, trg_mask):

        # Get the embedding for the src and trg tokens
        src_embeddings = self.src_embedding(src_tokens)
        trg_embeddings = self.trg_embedding(trg_tokens)

        # Add Positional Encoding
        src_embeddings = self.src_positional_encoding(src_embeddings)
        trg_embeddings = self.trg_positional_encoding(trg_embeddings)
        
        # Forward through the encoder
        src_embeddings = self.encoder(src_embeddings, src_mask)

        # Forward through the decoder
        # (B, T, d_model)
        trg_embeddings = self.decoder(trg_embeddings, src_embeddings, trg_mask, src_mask)
        
        # Project into vocab length and get log probabilities (for the kl div loss)
        # (B, T, d_model) -->  (B, T, vocab_length)
        trg_log_p = self.log_softmax(self.output_linear(trg_embeddings))

        return trg_log_p
    
        
class TransformerEncoder(nn.Module):
    """ N Layer Decoder With Masking """
    def __init__(self, encoder_layer, N, bias):
        super().__init__()
        assert isinstance(encoder_layer, TransformerEncoderLayer)

        # clone The decoder Layer N times
        self.encoder_layers = _get_clones(encoder_layer, N) 
        self.norm = LayerNorm(encoder_layer.d_model, bias)

    def forward(self, src, src_mask):

        # Forward through the encoder stack
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)
        
        return self.norm(src)
    
class TransformerEncoderLayer(nn.Module):
    """ Comprimised of self attention followed by a  feed forward """

    def __init__(self, d_model, self_attn, feed_forward, bias, dropout_p):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.norm1 = LayerNorm(d_model, bias=bias)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.norm2 = LayerNorm(d_model, bias=bias)
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, src, src_mask):
     
        src = self.norm1(src)
        src = self.self_attn(q=src, k=src, v=src, mask=src_mask)
        src = src + self.dropout1(src)

        src = self.norm2(src)
        src = self.feed_forward(src)
        src = src + self.dropout2(src)

        return src
    
class TransformerDecoder(nn.Module):
    """ N Layer Decoder With Masking """
    def __init__(self, decoder_layer, N, bias):
        super().__init__()
        assert isinstance(decoder_layer, TransformerDecoderLayer)
        
        # clone The decoder Layer N times
        self.decoder_layers = _get_clones(decoder_layer, N) 
        self.norm = LayerNorm(decoder_layer.d_model, bias)

    def forward(self, trgb, srcb, trg_mask, src_mask):

        # Forward through the decoder stack
        for decoder_layer in self.decoder_layers:
            trgb = decoder_layer(trgb, srcb, trg_mask, src_mask)

        return self.norm(trgb)
    

class TransformerDecoderLayer(nn.Module):
    """ Comprimised of Both self attention and cross attention followed by a feed forward """
    def __init__(self, d_model, multi_head_attention, feed_forward, bias, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = copy.deepcopy(multi_head_attention)
        self.src_attn = copy.deepcopy(multi_head_attention)
        self.feed_forward = feed_forward

        self.norm1 = LayerNorm(d_model, bias=bias)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.norm2 = LayerNorm(d_model, bias=bias)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.norm3 = LayerNorm(d_model, bias=bias)
        self.dropout3 = nn.Dropout(p=dropout_p)

    
    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.norm1(trg)
        trg = self.self_attn(q=trg, k=trg, v=trg, mask=trg_mask)
        trg = trg + self.dropout1(trg)

        trg = self.norm2(trg)
        trg = self.src_attn(q=trg, k=src, v=src, mask=src_mask)
        trg = trg + self.dropout2(trg)

        trg = self.norm3(trg)
        trg = self.feed_forward(trg)
        trg = trg + self.dropout3(trg)

        return trg
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, expansion_multiplier, bias, dropout_p=0.1):
        super().__init__()
        self.fc = nn.Linear(d_model, expansion_multiplier * d_model, bias=bias)
        # Original Paper Uses RELU instead
        self.gelu = nn.GELU()
        self.proj = nn.Linear(expansion_multiplier * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p) # Not in the original Paper
    
    def forward(self, x):
        # (batch size, max sequence length, d_model)
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias, dropout_p=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_size = d_model // n_heads 
        self.dropout_p = dropout_p
        self.bias = bias

        # Key, Query, Value Projection for all heads, but in a batch
        # self.attn_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=bias)
        self.linear_Q = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.linear_K = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.linear_V = nn.Linear(self.d_model, self.d_model, bias=bias)
        
        # Output Projections 
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)

        # Reguralization (Not in Original Paper)
        self.attn_dropout = nn.Dropout(self.dropout_p)
        self.res_dropout = nn.Dropout(self.dropout_p)

    def attention(self, q, k, v, mask=None):
        # Scaled dot-product attention
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        
        # Optionally Mask tokens by setting their values to -inf so that it will have value of 0 when it goes through
        # softmax, so that the attention pays no attention to these values (don't cheat from it)
        if mask is not None:
            if len(mask.shape) == 2: 
                # (B, T) --> (B, 1, 1,T)
                mask = mask.unsqueeze(1).unsqueeze(2)
            if len(mask.shape) == 3: 
                # (B, T, T) --> (B, 1, T, T)
                mask = mask.unsqueeze(1)
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        # Compute Attention
        scores = F.softmax(scores, dim=-1)

        # Apply Dropout (not on the original Paper)
        scores = self.attn_dropout(scores)

        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        return scores @ v

    def forward(self, q, k, v, mask=None):
        B, T, C = q.size() # (batch size, max sequence length, d_model)
   
        # Linear Projection 
        q = self.linear_Q(q)
        k = self.linear_K(k)
        v = self.linear_V(v)

        # Split into multiple Heads
        # (B, T, nh*hs) --view-> (B, T, nh, hs) --transpose->  (B, nh, T, hs) for each of q, k, v
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)   
        k = k.view(B, k.size(1), self.n_heads, self.head_size).transpose(1, 2) 
        v = v.view(B, v.size(1), self.n_heads, self.head_size).transpose(1, 2)

        # Compute Attention
        attention = self.attention(q, k, v, mask) # (B, nh, T, hs)

        # (B, nh, T, hs) --transpose-> (B, T, nh, hs) -> (B, T, nh*hs)
        # attention = attention.transpose(1, 2).contiguous().view(B, T, C) # Could incur additional matrix copy
        attention = attention.transpose(1, 2).reshape(B, -1, self.n_heads * self.head_size) # equivilant line

        # Output Projection & Dropout
        # (B, T, nh*hs) --> (B, T, nh*hs)
        return self.res_dropout(self.out_proj(attention))
    
class PositionalEncoding(nn.Module):
    """ PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) """
    
    def __init__(self, d_model, max_len, dropout_p):
        """ Computing the positional encoding once at initialization"""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        # represents the positions (pos) in the sequence
        positions = torch.arange(0, max_len).unsqueeze(1)

        # torch.arange(0, d_model, 2) generates a tensor of values, this tensor represents the values of 2i
        # in the term 10000^(2i/d_model). The calculated tensor is then divided by d_model to compute (2i/d_model)
        freqs = torch.pow(10000, -torch.arange(0, d_model, 2) / d_model)

        positional_encodings = torch.zeros(max_len, d_model)
        # Assigning Sin values to even dimensions
        positional_encodings[:, 0::2] = torch.sin(positions * freqs)
        # Assigning Cosine values to odd dimensions
        positional_encodings[:, 1::2] = torch.cos(positions * freqs)

        # Register Buffer is used to to save position encoding in the state_dict
        # and making them non trainable
        self.register_buffer('positional_encodings', positional_encodings)

    def forward(self, x):
        # batch of embeddings (x) size: (batch size, max sequence length, d_model)
        
        # Extracting (max sequence length, d_model) and adding it to x which broadcasts the batch dimension
        x = x + self.positional_encodings[:x.shape[1]]#.requires_grad_(False)

        # Applying Reguralization Using Dropout
        return self.dropout(x)
    

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()    
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)
    
class LayerNorm(nn.Module):
    """ LayerNorm with optional bias"""

    def __init__(self, features, bias, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        self.eps = eps

    def forward(self, input):
        """ https://arxiv.org/abs/1607.06450 """
        # Implementation without torch layer_norm 
        # mean = self.input.mean(-1,keepdim=True)
        # std = self.input.std(-1, keepdim=True)
        # out = self.weight * (self.input - mean) / (std + self.eps).sqrt() + self.bias
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _analyze_state_dict_shapes_and_names(model):
  """Prints detailed information about the model's state dictionary.

  This function iterates through the model's state dictionary and prints the names 
  and shapes of all parameters. It also raises an exception if any parameter is 
  not marked for training (requires_grad=False).

  """
  print("Keys of the model state_dict:")
  print(model.state_dict().keys())

  print("\nParameter details (name, shape):")
  for name, param in model.named_parameters():
    print(name, param.shape)
    if not param.requires_grad:
      raise Exception('Expected all parameters to be trainable. Found untrainable parameter:', name)


def run_transformer_test(src_vocab_size, trg_vocab_size, max_length, d_model, n_heads, n_layers, dropout_p, batch_size, src_seq_length=None, trg_seq_length=None):
  """Runs a single test case for the Transformer model with specified parameters.

  Args:
      src_vocab_size: Size of the source vocabulary.
      trg_vocab_size: Size of the target vocabulary.
      max_length: Maximum sequence length.
      d_model: Dimensionality of the model.
      n_heads: Number of attention heads.
      n_layers: Number of layers in both encoder and decoder.
      dropout_p: Dropout probability.
      batch_size: Batch size for the test case.
      src_seq_length: Optional source sequence length (defaults to max_length).
      trg_seq_length: Optional target sequence length (defaults to max_length).
  """
  transformer = Transformer(
      src_vocab_length=src_vocab_size,
      trg_vocab_length=trg_vocab_size,
      max_length=max_length,
      d_model=d_model,
      n_heads=n_heads,
      n_layers=n_layers,
      dropout_p=dropout_p
  )

  # Use provided sequence lengths or default to max_length
  src_seq_length = src_seq_length or max_length
  trg_seq_length = trg_seq_length or max_length

  src_token_ids = torch.randint(0, src_vocab_size, (batch_size, src_seq_length))
  trg_token_ids = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_length))

  # Create masks with proper dimensions
  src_mask = (src_token_ids != 0).unsqueeze(1).unsqueeze(2)
  trg_mask = (trg_token_ids != 0).unsqueeze(1).unsqueeze(2)
  trg_mask = trg_mask & torch.tril(torch.ones((batch_size, 1, trg_seq_length, trg_seq_length), device=trg_token_ids.device)).type_as(trg_mask)

  # Forward pass
  outputs = transformer(src_token_ids, trg_token_ids, src_mask, trg_mask)

  # Check output dimensions and probabilities
  print(f"Test (src_vocab: {src_vocab_size}, trg_vocab: {trg_vocab_size}, seq_len: {src_seq_length}/{trg_seq_length}):")
  print("Output Log-Probabilities Shape:", outputs.shape)  # Expected: (batch_size * trg_seq_length, trg_vocab_size)
  prob_sum = torch.exp(outputs).view(batch_size, trg_seq_length, -1).sum(dim=-1)
  print("Sum of probabilities for each token position (should be close to 1):", prob_sum.mean(dim=1))
  del transformer, src_token_ids, trg_token_ids, src_mask, trg_mask, prob_sum, outputs
        
if __name__ == "__main__":
  BASELINE_MODEL_CONFIG = {
    "num_layers": 6,
    "d_model": 512,
    "num_heads": 8,
    "dropout_prob": 0.1,
    "label_smoothing_value": 0.1,
  }

  BIG_MODEL_CONFIG = {
    "num_layers": 6,
    "d_model": 1024,
    "num_heads": 16,
    "dropout_prob": 0.3,
    "label_smoothing_value": 0.1,
  }

  use_big_transformer = True
  model_config = BIG_MODEL_CONFIG if use_big_transformer else BASELINE_MODEL_CONFIG

  # Define dummy data 
  src_vocab_size = 11
  trg_vocab_size = 11
  max_length = 50000
  src_tokens = torch.randint(1, 10, size=(3, 2))
  trg_tokens = torch.randint(1, 10, size=(3, 2))

  # Create and analyze the transformer model
  transformer = Transformer(
      src_vocab_length=src_vocab_size,
      trg_vocab_length=trg_vocab_size,
      max_length=max_length,
      d_model=model_config["d_model"],
      n_heads=model_config["num_heads"],
      n_layers=model_config["num_layers"],
      dropout_p=model_config["dropout_prob"]
  )

  _analyze_state_dict_shapes_and_names(transformer)
  print(f'Size of the {"big" if use_big_transformer else "baseline"} transformer = {_count_parameters(transformer)}')

  # Perform a forward pass
  out = transformer(src_tokens, trg_tokens, src_mask=None, trg_mask=None)

  run_transformer_test(src_vocab_size, trg_vocab_size, max_length, model_config["d_model"], model_config["num_heads"], model_config["num_layers"],
                        model_config["dropout_prob"], batch_size=8, src_seq_length=4, trg_seq_length=4)
