import os
import random
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, label_file, feat_dir, vocabulary):
        self.feat_dir = feat_dir
        self.vocabulary = vocabulary
        with open(label_file, 'r') as f:
            self.training_labels = json.load(f)

    def __len__(self):
        return len(self.training_labels)

    def __getitem__(self, idx):
        video_id = self.training_labels[idx]["id"]
        captions = self.training_labels[idx]["caption"]

        # Strategy: Randomly select one caption for training.
        caption = random.choice(captions)

        # Tokenize and convert to indices
        tokens = self.vocabulary.tokenize(caption)

        vocab = self.vocabulary.word_dict
        caption_indices = [vocab['<BOS>']] + \
                          [vocab.get(word, vocab['<UNK>']) for word in tokens] + \
                          [vocab['<EOS>']]

        filename = os.path.join(self.feat_dir, video_id + ".npy")
        try:
            feat = np.load(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # Return a zero tensor of correct shape if loading fails (safety fallback)
            feat = np.zeros((80, 4096))

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(caption_indices, dtype=torch.long)