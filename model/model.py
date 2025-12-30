import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Linear layer to project video features (4096) to the embedding dimension
        self.video_fc = nn.Linear(4096, embed_dim)
        
        # Layer 1: Takes video features
        self.lstm_layer1 = nn.LSTM(input_size=embed_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   dropout=dropout)
                                   
        # Layer 2: Takes output of Layer 1 concatenated with padding text
        self.lstm_layer2 = nn.LSTM(input_size=hidden_dim + embed_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   dropout=dropout)
        
    def forward(self, features):
        """
        Args:
            features: (batch_size, 80, 4096) - Video features
        Returns:
            (h1, c1), (h2, c2): Final hidden and cell states for both layers
        """
        batch_size = features.size(0)
        video_len = features.size(1)
        device = features.device
        
        # Project video features: (batch, 80, embed_dim)
        video_embed = self.video_fc(features)
        
        # Padding for text input during video phase (batch, 1, embed_dim)
        padding_text = torch.zeros(batch_size, 1, self.embed_dim).to(device)
        
        h1, c1 = None, None
        h2, c2 = None, None
        
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
            
        return (h1, c1), (h2, c2)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pretrained_embedding=None):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer for text tokens
        if pretrained_embedding is not None:
            if pretrained_embedding.shape != (vocab_size, embed_dim):
                raise ValueError("Shape of pretrained_embedding does not match vocab_size and embed_dim.")
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            
        # Layer 1: Takes padding video
        self.lstm_layer1 = nn.LSTM(input_size=embed_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   dropout=dropout)
                                   
        # Layer 2: Takes output of Layer 1 concatenated with text embedding
        self.lstm_layer2 = nn.LSTM(input_size=hidden_dim + embed_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   dropout=dropout)
                                   
        # Linear layer to project hidden state to vocabulary size for prediction
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, captions, encoder_states, teacher_forcing_ratio=1.0):
        """
        Args:
            captions: (batch_size, max_len) - Caption token indices
            encoder_states: tuple of ((h1, c1), (h2, c2)) from encoder
            teacher_forcing_ratio: float
        """
        batch_size = captions.size(0)
        max_len = captions.size(1)
        device = captions.device
        
        # Unpack encoder states
        (h1, c1), (h2, c2) = encoder_states
        
        # Padding for video input during caption phase (batch, 1, embed_dim)
        padding_video = torch.zeros(batch_size, 1, self.embed_dim).to(device)
        
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(device)
        
        # --- Phase 2: Decoding Caption ---
        # First input is <BOS>
        decoder_input = captions[:, 0]
        
        for t in range(1, max_len):
            # Layer 1 Input: Padding Video (since no more frames)
            out1, (h1, c1) = self.lstm_layer1(padding_video, (h1, c1))
            
            # Embed current decoder input
            txt_embed = self.embedding(decoder_input).unsqueeze(1)
            
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

class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pretrained_embedding=None, freeze=False):
        super(LSTM_Model, self).__init__()
        self.encoder = Encoder(embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout, pretrained_embedding)
        
    def forward(self, features, captions, teacher_forcing_ratio=1.0):
        """
        Args:
            features: (batch_size, 80, 4096) - Video features
            captions: (batch_size, max_len) - Caption token indices
        """
        encoder_states = self.encoder(features)
        outputs = self.decoder(captions, encoder_states, teacher_forcing_ratio)
        return outputs