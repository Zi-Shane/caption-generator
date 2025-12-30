import torch
import torch.nn as nn
import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from .model import LSTM_Model

class Trainer:
    def __init__(self, config, train_dataset, val_dataset, vocabulary, device):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.vocabulary = vocabulary
        
        # We need a map for fast lookup of captions by video_id during validation
        # self.val_dataset.training_labels is a list, so we build a dict here.
        self.val_captions_map = {item['id']: item['caption'] for item in self.val_dataset.training_labels}
        
        self.vocab_size = len(self.vocabulary.word_dict)
        self.device = device
        self.embed_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.save_interval = config['save_interval']
        self.max_checkpoints = config['max_checkpoints']
        self.checkpoint_dir = config['checkpoint_dir']
        self.model = LSTM_Model(self.vocab_size, self.embed_dim, self.hidden_dim, self.num_layers, self.dropout_rate).to(device)

        # Define optimizer and criterion directly inside the class
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming <PAD> is index 0

        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn # We'll define a collate_fn for padding sequences
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        # Pad captions to the maximum length in the batch
        # Features are assumed to be of fixed size (80, 4096)
        features = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        video_ids = [item[2] for item in batch]

        # Find max caption length
        max_len = max(len(cap) for cap in captions)

        # Pad all captions to max_len
        padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
        for i, cap in enumerate(captions):
            padded_captions[i, :len(cap)] = cap

        return features, padded_captions, video_ids

    def _train_epoch(self, epoch):
        self.model.train() # Set model to training mode
        total_loss = 0
        teacher_forcing_ratio = 0.5 if epoch > 2 else 1.0
        for i, (features, captions, _) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
            features = features.to(self.device)
            captions = captions.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features, captions, teacher_forcing_ratio)

            # Remove <PAD> tokens from target for loss calculation
            outputs = outputs[:, 1:].reshape(-1, self.model.vocab_size)
            captions = captions[:, 1:].reshape(-1)

            loss = self.criterion(outputs, captions)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval() # Set model to evaluation mode
        total_val_loss = 0
        references = []
        hypotheses = []
        
        with torch.no_grad(): # Disable gradient calculations during validation
            for i, (features, captions, video_ids) in enumerate(self.val_loader):
                features = features.to(self.device)
                captions = captions.to(self.device)

                outputs = self.model(features, captions, teacher_forcing_ratio=0.0) # No teacher forcing during validation

                # Calculate loss (using reshape as before)
                outputs_flat = outputs[:, 1:].reshape(-1, self.model.vocab_size)
                captions_flat = captions[:, 1:].reshape(-1)

                loss = self.criterion(outputs_flat, captions_flat)
                total_val_loss += loss.item()
                
                # BLEU score calculation
                # Get predicted words
                preds = outputs.argmax(dim=2) # (batch, max_len)
                
                for j, vid in enumerate(video_ids):
                    pred_indices = preds[j].tolist()
                    pred_words = []
                    for idx in pred_indices:
                        if idx == self.vocabulary.word_dict['<EOS>']:
                            break
                        if idx not in [self.vocabulary.word_dict['<PAD>'], self.vocabulary.word_dict['<BOS>']]:
                            word = self.vocabulary.rev_word_dict.get(idx, '<UNK>')
                            pred_words.append(word)
                    
                    hypotheses.append(pred_words)
                    
                    # Get references
                    raw_refs = self.val_captions_map[vid]
                    ref_list = [self.vocabulary.tokenize(r) for r in raw_refs]
                    references.append(ref_list)

        avg_val_loss = total_val_loss / len(self.val_loader)
        bleu_score = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, BLEU-1: {bleu_score:.4f}")
        return avg_val_loss

    def save_checkpoint(self, epoch, path, best_loss):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_loss
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            avg_train_loss = self._train_epoch(epoch)
            print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

            avg_val_loss = self.validate(epoch)

            # Save checkpoint if current validation loss is the best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, os.path.join(self.checkpoint_dir, 'best_model.pth'), best_val_loss)
