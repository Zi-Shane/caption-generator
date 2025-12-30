import torch
import sys
import os

# Add current directory to path so we can import model
sys.path.append(os.getcwd())

from model.model import LSTM_Model

def verify():
    vocab_size = 100
    embed_dim = 32
    hidden_dim = 64
    num_layers = 1
    dropout = 0.5
    batch_size = 2
    max_len = 10
    features_dim = 4096

    print("Initializing LSTM_Model...")
    model = LSTM_Model(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    print("Model initialized.")

    print("Creating dummy inputs...")
    features = torch.randn(batch_size, 80, features_dim)
    captions = torch.randint(0, vocab_size, (batch_size, max_len))

    print("Running forward pass...")
    output = model(features, captions)
    print("Output shape:", output.shape)
    
    expected_shape = (batch_size, max_len, vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("Verification Passed!")

if __name__ == "__main__":
    verify()
