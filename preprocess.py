import os
import json
import random

def split_training_data(original_label_file, output_train_file, output_val_file, split_ratio=0.8, seed=42):
    """
    Splits the original training label file into training and validation sets.

    Args:
        original_label_file (str): Path to the original JSON file containing all training labels.
        output_train_file (str): Path to save the new training label JSON file.
        output_val_file (str): Path to save the new validation label JSON file.
        split_ratio (float): The proportion of data to use for training (e.g., 0.8 for 80%).
        seed (int): Seed for random shuffling to ensure reproducibility.

    """
    with open(original_label_file, 'r') as f:
        all_data = json.load(f)

    random.seed(seed)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * split_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    os.makedirs(os.path.dirname(output_train_file) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_val_file) or '.', exist_ok=True)

    with open(output_train_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(output_val_file, 'w') as f:
        json.dump(val_data, f, indent=4)

    print(f"Successfully split data: {len(train_data)} for training, {len(val_data)} for validation.")
    print(f"Training labels saved to: {output_train_file}")
    print(f"Validation labels saved to: {output_val_file}")
    print("--------------------------------")