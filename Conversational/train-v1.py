import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameter
vocab_size = 10000
d_model = 512
n_heads = 8
dim_feedforward = 2048
num_layers = 6
num_epochs = 10
learning_rate = 0.001

# Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        pe = torch.zeros(1000, d_model)
        position = torch.arange(0, 1000, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = self.combine_heads(scaled_attention)
        output = self.out_linear(scaled_attention)
        return output

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_length, self.d_model)

# Feedforward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward):
        super(TransformerBlock, self).__init__()
        self.self_attention = SelfAttention(d_model, n_heads)
        self.feedforward = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output = self.self_attention(x)
        x = x + attention_output
        x = self.norm1(x)
        feedforward_output = self.feedforward(x)
        x = x + feedforward_output
        x = self.norm2(x)
        return x

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, dim_feedforward, num_layers):
        super(Transformer, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dim_feedforward) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.fc(x)
        return x

# Trainingsdaten
text = "Das ist ein Beispieltext."
tokens = text.split()

# Vokabular erstellen
vocab = list(set(tokens))
vocab_size = len(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

# Tokenisierung
token_ids = [word2idx[word] for word in tokens]

# Modell initialisieren
model = Transformer(vocab_size, d_model, n_heads, dim_feedforward, num_layers)

# Loss-Funktion und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trainingsschleife
for epoch in range(num_epochs):
    optimizer.zero_grad()
    inputs = torch.tensor(token_ids[:-1], dtype=torch.long)
    targets = torch.tensor(token_ids[1:], dtype=torch.long)
    outputs = model(inputs.unsqueeze(0))
    loss = criterion(outputs.squeeze(0), targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Beispielgenerierung
input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
generated_ids = model(input_ids.unsqueeze(0)).argmax(dim=-1)
generated_text = [idx2word[idx.item()] for idx in generated_ids.squeeze()]
print("Generated Text:", " ".join(generated_text))
