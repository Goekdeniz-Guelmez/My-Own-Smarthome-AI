import torch
import torch.nn as nn
import torch.optim as optim

# Lade den conversationalen Datensatz aus der Textdatei
with open("Goku-bot/My-Own-Smarthome-AI/Conversational/dataset.txt", "r", encoding="utf-8") as file:
    data = file.readlines()
data = [line.strip() for line in data]

# Tokenisierung des Datensatzes
tokenizer = YourTokenizer()  # Ersetze "YourTokenizer" durch den tatsächlichen Tokenizer deiner Wahl

tokenized_data = []
for line in data:
    tokens = tokenizer.tokenize(line)
    tokenized_data.append(tokens)

# Vokabular erstellen
all_tokens = [token for dialog in tokenized_data for token in dialog]
vocab = list(set(all_tokens))
vocab_size = len(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

# Konvertiere den Datensatz in Token-IDs
token_ids = [[word2idx[token] for token in dialog] for dialog in tokenized_data]

# Hyperparameter
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

# Self-Attention Layer, Feedforward Layer und Transformer Block bleiben unverändert

# Transformer Modell
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

# Modell initialisieren
model = Transformer(vocab_size, d_model, n_heads, dim_feedforward, num_layers)

# Loss-Funktion und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trainingsschleife
for epoch in range(num_epochs):
    total_loss = 0
    for dialog in token_ids:
        optimizer.zero_grad()
        inputs = torch.tensor(dialog[:-1], dtype=torch.long)
        targets = torch.tensor(dialog[1:], dtype=torch.long)
        outputs = model(inputs.unsqueeze(0))
        loss = criterion(outputs.squeeze(0), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch+1}, Loss: {total_loss}")

# Beispielgenerierung
dialog = "Benutzer: Hallo!\n"
input_ids = [word2idx[token] for token in dialog.split()]
input_ids = torch.tensor(input_ids, dtype=torch.long)
generated_ids = model(input_ids.unsqueeze(0)).argmax(dim=-1)
generated_text = [idx2word[idx.item()] for idx in generated_ids.squeeze()]
print("Generated Text:", " ".join(generated_text))
