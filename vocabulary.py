import json
import re
import numpy as np
from tqdm import tqdm
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.vocab = {}  # word -> index
        self.rev_vocab = {}  # index -> word
        self.embedding_matrix = None # To store the loaded word embeddings
        self.initialized_randomly = None
        self.glove_sizes = 400000 # Number of Rows 

    def tokenize(self, sentence):
        """
        Helper function to tokenize a sentence.
        Converts to lowercase, removes punctuation, and splits by space.
        """
        sentence = re.sub(r'[^\w\s-]', '', sentence).lower() # Adjusted regex for wider char support, if needed
        return sentence.split()

    def build_vocab(self, file_path, min_count=1):
        # Read data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Get all caption list
        all_captions = []
        for item in data:
            all_captions.extend(item['caption'])

        # Initialize with special tokens
        self.vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.rev_vocab = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        idx = 4

        # Count words first
        counter = Counter()
        for caption in tqdm(all_captions, desc="Counting words"):
            caption = self.tokenize(caption)
            counter.update(caption)

        # Build vocabulary
        for word, count in counter.items():
            if count >= min_count:
                self.vocab[word] = idx
                self.rev_vocab[idx] = word
                idx += 1

        print(f"Vocabulary built with {len(self.vocab)} unique words (min_count={min_count}).")

    def load_word_vector(self, file_path, embed_dim=200):
        # Initialize embedding matrix with zeros
        # Use np.zeros to ensure it's float32
        self.embedding_matrix = np.zeros((len(self.vocab), embed_dim), dtype=np.float32)
        found_words = set()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading word vectors", total=self.glove_sizes):
                values = line.split()
                if not values: # Skip empty lines
                    continue
                word = values[0]

                # Check if the remaining values match the expected embedding dimension
                if len(values) - 1 != embed_dim:
                    # print(f"Warning: Skipping word '{word}' due to dimension mismatch. Expected {embed_dim}, got {len(values)-1}.")
                    continue

                try:
                    vector = np.asarray(values[1:], dtype='float32')
                except ValueError:
                    # print(f"Warning: Skipping word '{word}' due to non-numeric vector values.")
                    continue

                if word in self.vocab:
                    self.embedding_matrix[self.vocab[word]] = vector
                    found_words.add(word)

        # Handle words in our vocabulary that were not found in the pre-trained vectors
        self.initialized_randomly = [] # Reset for consistent behavior
        for word, idx in self.vocab.items():
            if word not in found_words:
                # Initialize unknown words with random uniform vectors, a common practice
                self.embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embed_dim).astype(np.float32)
                self.initialized_randomly.append(idx)

        print(f"Successfully loaded {len(found_words)} word vectors from {file_path}.")
        print(f"Initialized {[self.rev_vocab[i] for i in self.initialized_randomly]} words in vocabulary randomly.")
        print(f"Embedding matrix shape: {self.embedding_matrix.shape}")