import io
from logging import warning
from typing import Union, List
from site import PREFIXES
import warnings
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import matplotlib.pyplot as plt
import copy

import torch
from transformers import AutoTokenizer
import random

class HybridRetrievalDataset:
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

    def tokenize_prompts(self, prompts, prepend_bos=False):
        tokenized_output = self.tokenizer(prompts, padding=True, return_tensors="pt")
        self.toks = tokenized_output.input_ids
        return self.toks

    def corrupt_prompt(self, prompt, capitals):
        # Replace 'washington' with a random city from the list of capitals
        corrupted_prompt = prompt.replace('washington', random.choice(capitals))
        return corrupted_prompt

    def print_prompts(self, clean_prompts, corrupted_prompts):
        print("Clean Prompts:")
        for prompt in clean_prompts:
            print(prompt)
        print("\nCorrupted Prompts:")
        for prompt in corrupted_prompts:
            print(prompt)

    def get_dataset(self, factual=True):
        # Factual prompts
        factual_prompts = [
            "alice lives in france, paris - alice, bob lives in germany, berlin - bob, john lives in usa, washington - john",
            "alice lives france, paris - alice, bob lives germany, berlin - bob, john lives USA, washington - john",
            "alice resides france, paris - alice, bob resides germany, berlin - bob, john resides usa, washington - john",
            "alice resides in france, paris - alice, bob resides in germany, berlin - bob, john resides in usa, washington - john",
            "alice works france, paris - alice, bob works germany, berlin - bob, john works usa, washington - john",
            "alice loves france, paris - alice, bob loves germany, berlin - bob, john loves usa, washington - john",
            "alice travels france, paris - alice, bob travels germany, berlin - bob, john travels usa, washington - john",
            "alice is french, paris - alice, bob is german, berlin - bob, john is american, washington - john",
            "alice is from france, paris - alice, bob is from germany, berlin - bob, john is from usa, washington - john"
        ]
        
        # Counterfactual prompts
        counterfactual_prompts = [
            "alice drinks france, paris - alice, bob drinks germany, berlin - bob, john drinks usa, washington - john",
            "alice cuts france, paris - alice, bob cuts germany, berlin - bob, john cuts usa, washington - john",
            "alice drills france, paris - alice, bob drills germany, berlin - bob, john drills usa, washington - john"
        ]

        # Combine all prompts
        all_prompts = factual_prompts + counterfactual_prompts

        # Tokenize the prompts
        clean_data = self.tokenize_prompts(all_prompts)

        # List of 12 capitals to replace 'Washington'
        capitals = [
            "london", "tokyo", "bucharest", "berlin", "rome", 
            "madrid", "beijing", "moscow", "cairo", "ottawa", 
            "canberra", "sofia"
        ]

        # Create corrupted prompts by replacing 'washington' with a random capital
        corrupted_prompts = [self.corrupt_prompt(prompt, capitals) for prompt in all_prompts]
        corrupted_data = self.tokenize_prompts(corrupted_prompts)

        # Print clean and corrupted prompts
        self.print_prompts(all_prompts, corrupted_prompts)

        return clean_data, corrupted_data

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hybrid_retrieval_dataset = HybridRetrievalDataset()
    
    # Get datasets
    clean_data, corrupted_data = hybrid_retrieval_dataset.get_dataset(factual=True)
    clean_data = clean_data.to(device)
    corrupted_data = corrupted_data.to(device)

    # Print the tokenized datasets for verification
    print("Clean Data Datasets:")
    print(clean_data)

    print("\nCorrupted Data Datasets:")
    print(corrupted_data)

    # Define sequence length and number of examples
    seq_len = clean_data.size(1)
    num_examples = clean_data.size(0) // 2

    # Create the validation and test splits
    default_data = clean_data[:num_examples*2, :seq_len - 1].to(device)
    patch_data = corrupted_data[:num_examples*2, :seq_len - 1].to(device)
    labels = clean_data[:num_examples*2, seq_len - 1].to(device)
    wrong_labels = torch.as_tensor(clean_data[:num_examples*2, seq_len - 1], dtype=torch.long, device=device)

    # Split into validation and test sets
    validation_data = default_data[:num_examples, :]
    validation_patch_data = patch_data[:num_examples, :]
    validation_labels = labels[:num_examples]
    validation_wrong_labels = wrong_labels[:num_examples]

    test_data = default_data[num_examples:, :]
    test_patch_data = patch_data[num_examples:, :]
    test_labels = labels[num_examples:]
    test_wrong_labels = wrong_labels[num_examples:]

    # Print shapes for verification
    print(f"Shape of validation_data: {validation_data.shape}")
    print(f"Shape of validation_patch_data: {validation_patch_data.shape}")
    print(f"Shape of validation_labels: {validation_labels.shape}")
    print(f"Shape of validation_wrong_labels: {validation_wrong_labels.shape}")

    print(f"Shape of test_data: {test_data.shape}")
    print(f"Shape of test_patch_data: {test_patch_data.shape}")
    print(f"Shape of test_labels: {test_labels.shape}")
    print(f"Shape of test_wrong_labels: {test_wrong_labels.shape}")