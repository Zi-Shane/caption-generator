import random
import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer for text tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Linear layer to project video features (4096) to the embedding dimension
        self.video_fc = nn.Linear(4096, embed_dim)

        # Layer 1: Takes video features (or padding)
        self.lstm_layer1 = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        # Layer 2: Takes output of Layer 1 concatenated with text embedding (or padding)
        # Input size = hidden_dim (from layer 1) + embed_dim (from text)
        self.lstm_layer2 = nn.LSTM(input_size=hidden_dim + embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        # Linear layer to project hidden state to vocabulary size for prediction
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def load_pretrained_embeddings(self, embedding_matrix, unfreeze_ids=None, freeze_embeddings=True):
        """
        Args:
            embedding_matrix (np.array): A numpy array of shape (vocab_size, embed_dim)
                                         containing the pre-trained word vectors.
            unfreeze_ids (list): A list of integer indices for words
                                                 that were initialized randomly and should NOT be frozen.
            freeze_embeddings (bool): If True, the embedding layer weights will be frozen,
                                      with exceptions for `unfreeze_ids`.
        """
        if embedding_matrix.shape != self.embedding.weight.shape:
            raise ValueError(
                f"Pre-trained embedding matrix shape {embedding_matrix.shape} "
                f"does not match model embedding layer shape {self.embedding.weight.shape}."
            )

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = freeze_embeddings

        print(f"Loaded pre-trained embeddings of shape {embedding_matrix.shape}. ")
        print(f"Frozen: {freeze_embeddings}. Individual weights frozen selectively based on `unfreeze_ids` if provided.")


    def forward(self, features, captions, teacher_forcing_ratio=0.5):
        """
        Args:
            features: (batch_size, 80, 4096) - Video features
            captions: (batch_size, max_len) - Caption token indices
        """
        batch_size = features.size(0)
        max_len = captions.size(1)
        video_len = features.size(1)
        device = features.device

        # Project video features: (batch, 80, embed_dim)
        video_embed = self.dropout(self.video_fc(features))

        # Initialize hidden states (automatically zeros if not provided)
        # We need to maintain states across the loop
        h1, c1 = None, None
        h2, c2 = None, None

        # Padding for text input during video phase (batch, 1, embed_dim)
        padding_text = torch.zeros(batch_size, 1, self.embed_dim).to(device)

        # Padding for video input during caption phase (batch, 1, embed_dim)
        padding_video = torch.zeros(batch_size, 1, self.embed_dim).to(device)

        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(device)

        # --- Phase 1: Encoding Video ---
        for t in range(video_len):
            # Layer 1 Input: Video frame t
            # Shape: (batch, 1, embed_dim)
            inp1 = video_embed[:, t, :].unsqueeze(1)
            out1, (h1, c1) = self.lstm_layer1(inp1, (h1, c1) if h1 is not None else None)

            # Layer 2 Input: Output of Layer 1 + Padding Text
            # Shape: (batch, 1, hidden_dim + embed_dim)
            inp2 = torch.cat((out1, padding_text), dim=2)
            out2, (h2, c2) = self.lstm_layer2(inp2, (h2, c2) if h2 is not None else None)

        # --- Phase 2: Decoding Caption ---
        # First input is <BOS>
        decoder_input = captions[:, 0]

        for t in range(1, max_len):
            # Layer 1 Input: Padding Video (since no more frames)
            out1, (h1, c1) = self.lstm_layer1(padding_video, (h1, c1))

            # Embed current decoder input
            txt_embed = self.dropout(self.embedding(decoder_input)).unsqueeze(1)

            # Layer 2 Input: Output of Layer 1 + Embedded Text
            inp2 = torch.cat((out1, txt_embed), dim=2)
            out2, (h2, c2) = self.lstm_layer2(inp2, (h2, c2))

            # Predict next word
            prediction = self.fc_out(out2.squeeze(1))
            outputs[:, t, :] = prediction

            # Teacher Forcing
            if random.random() < teacher_forcing_ratio:
                decoder_input = captions[:, t]
            else:
                decoder_input = prediction.argmax(1)

        return outputs