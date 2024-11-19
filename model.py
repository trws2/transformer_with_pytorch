import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # position encoding matrix of size seq_len x d_model (cols x rows)
        pe = torch.zeros(seq_len, d_model, device='mps', dtype=torch.float)

        # a vector (or matrix) of size 1 x seq_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # 1 / 10000^(2i/d_model) =
        #   exp(log(1 / 10000^(2i/d_model))) =
        #   exp(log(1) - log(10000^(2i/d_model))) =
        #   exp(0 - (2i/d_model) * log(10000)) =
        #   exp(2i * (-log(10000)/d_model))
        #
        # and we need to do this for each i
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # fill even entries of each position's embedding. positions will be go up to the sequence length
        pe[:, 0::2] = torch.sin(position * div_term)
        # fill odd entries of each position's embedding. positions will be go up to the sequence length
        pe[:, 1::2] = torch.cos(position * div_term)

        # unsqueeze because we have a batch of sentences. So the 1st dimension is reserved for different sentences.
        # The 1st dimension size is what we call batch_size
        # pe becomes 1 x seq_len x d_model
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # requires_grad_ is used to make this particular tensor `self.pe[:, :x.shape[1], :].requires_grad_(False)` not learned
        # as position encoding is statically defined, not learned via sgd.
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float=10**-6) -> None:
        # features is the size of last dimension of x of forward method below
        super().__init__()
        self.eps = 10**-6
        self.alpha = nn.Parameter(torch.ones(features, device='mps', dtype=torch.float))  # alpha is learnable multiplier
        self.bias = nn.Parameter(torch.zeros(features, device='mps', dtype=torch.float))  # alpha is learnable bias

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size] or [batch_size, seq_len, features] from __init__ method above
        #
        # last dim has the embedding, so compute mean and std at last (or -1) dimension
        # with keepdim == True, the shape of mean and std is (batch_size, sequence_length, 1)
        # with keepdim == False, the shape of mean and std is (batch_size, sequence_length)
        # so we set it True for layer norm computation below to allow broadcasting to happen.
        # See https://stackoverflow.com/questions/51371070/how-does-pytorch-broadcasting-work for
        # how broadcasting works.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # (x - mean) / (std + self.eps): [batch_size, seq_len, hidden_size]
        # self.alpha: [hidden_size], then broadcast to [batch_size, seq_len, hidden_size]
        # self.bias: [hidden_size], then broadcast to [batch_size, seq_len, hidden_size]
        # element x_i in a feature vector x apply the same alpha_i and bias_i, this is the
        # same for all [batch_size x seq_len feature vectors.
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        # this forward module apply 1 layer MLP with ReLU activation function with output size being d_ff
        # it then applies another linear layer (not MLP), to get a new feature representation with the same size as
        # input feature size (d_model). So essentially, d_ff represent the number of hidden units to learn.
        # note that dropout is only applied to the output MLP, not the output of 2nd linear layer.
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2
        self.dropout = nn.Dropout(dropout)

    # x: [batch_size, seq_len, feature_len], feature_len here is embedding or output_size from last module, which is d_model
    # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ResidualConnection(nn.Module):

    # features should ideally be named as feature_size or feature_dim
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()

        # a bit interesting fact about this residual connection is that instead of directly applying to sublayer,
        # we need to first apply layer norm then the sublayer, then dropout before we add to identify.
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.h = h

        assert d_model % h == 0, "d_model must be a multiplier of h"
        self.d_k = d_model // h
        self.d_model = d_model

    def attention(self, query, key, value, mask):
        # attention_mat: [batch_size, h, seq_len, seq_len]
        attention_mat = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_mat.masked_fill_(mask == 0, 1e-9)
        attention_mat = attention_mat.softmax(dim=-1)

        # the original implementation may have a problem that dropout will randomly zeros out
        # elements in attention_mat and scales up remaining ones by 1/(1-dropout_rate).
        # so the following dropout code may break property that the last dimension in attention_mat
        # need to be summed to 1.0. it looks to me that it is a bug.
        if self.dropout is not None:
            attention_mat = self.dropout(attention_mat)

        # [batch_size, h, seq_len, seq_len] x [batch_size, h, seq_len, d_k] -->
        # [batch_size, h, seq_len, d_k]
        # This matrix multiplication can be illustrated via the following example:
        #
        # A: 2 x 2 x 3
        # A = tensor([[[0.1000, 0.6000, 0.3000],
        #          [0.2000, 0.3000, 0.5000]],
        #
        #         [[0.5000, 0.4000, 0.1000],
        #          [0.3000, 0.4000, 0.3000]]])
        #
        # B: 2 x 3 x 4
        # B = tensor([[[1., 0., 1., 0.],
        #          [0., 1., 0., 1.],
        #          [1., 1., 1., 1.]],
        #
        #         [[0., 1., 1., 0.],
        #          [1., 0., 0., 1.],
        #          [0., 0., 1., 1.]]])
        #
        # C = A @ B: 2 x 2 x 4
        # C = tensor([[[0.4000, 0.9000, 0.4000, 0.9000],
        #          [0.7000, 0.8000, 0.7000, 0.8000]],
        #
        #         [[0.4000, 0.5000, 0.6000, 0.5000],
        #          [0.4000, 0.3000, 0.6000, 0.7000]]])
        #
        # C[0, 0] = 0.1000 * 1. + 0.6000 * 0. + 0.3000 * 1. = 0.4000
        # C[0, 1] = 0.1000 * 0. + 0.6000 * 1. + 0.3000 * 1. = 0.9000
        # C[1, 2] = 0.2000 * 1. + 0.3000 * 0. + 0.5000 * 1. = 0.8000
        return (attention_mat @ value), attention_mat

    def forward(self, q, k, v, mask):
        query = self.Wq(q)
        key = self.Wq(k)
        value = self.Wq(v)

        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, attention_mat = self.attention(query, key, value, mask)
        # 1. transpose a tensor does not return a new contiguous block of memory
        # but reshape operation via view requires the tensor being in contiguous block of memory
        # so we need contiguous() here after transpose.
        # 2. use -1 so that size of the 2nd dimension can be automatically infered based on the size of total size of the
        # tensor. Though I think in our case, we can directly use x.shape[0] instead.
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.Wo(x)


class EncoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        # in translational task, produce weighted embedding average from tokens in original language
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = [
            ResidualConnection(features, dropout) for _ in range(3)
        ]

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # in translational task, produce weighted embedding average from tokens generated from decoder in another language
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # in translational task, produce weighted embedding average from tokens in original language
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # input x: [batch_size, seq_len, d_model]
        x = self.proj(x)
        # x after self.proj: [batch_size, seq_len, vocab_size]
        #
        # why there is no softmax along the last dimension?
        # or it will be encoded in training loss and decoding function, so we do not really need softmax when defining the model arch
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embeds: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_embeds: InputEmbeddings,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeds = src_embeds
        self.src_pos = src_pos
        self.tgt_embeds = tgt_embeds
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        x = self.src_embeds(src)                
        x = self.src_pos(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):        
        x = self.tgt_embeds(tgt)
        x = self.tgt_pos(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    dropout: float = 0.1,
    N: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
) -> Transformer:
    # create raw embeddings
    src_embeds = InputEmbeddings(d_model, src_vocab_size)
    tgt_embeds = InputEmbeddings(d_model, tgt_vocab_size)

    # add positional embeddings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # initialize encoder blocks
    encoder_blocks = nn.ModuleList()
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            self_attention_block, feed_forward_block, d_model, dropout
        )
        encoder_blocks.append(encoder_block)
    encoder = Encoder(d_model, encoder_blocks)

    # initialize decoder blocks
    decoder_blocks = nn.ModuleList()
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            self_attention_block,
            cross_attention_block,
            feed_forward_block,
            d_model,
            dropout,
        )
        decoder_blocks.append(decoder_block)
    decoder = Decoder(d_model, decoder_blocks)

    # create project layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create transformer
    transformer = Transformer(
        encoder, decoder, src_embeds, src_pos, tgt_embeds, tgt_pos, projection_layer
    )

    # initialize parameter of transformer
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
