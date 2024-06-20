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
    def __init__(self, prompts=None, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # self.toks = torch.Tensor(self.tokenizer(prompts, padding=True).input_ids).type(
        #     torch.int
        # )

    def tokenize_prompts(self, prompts: str):
        tokenized_output = self.tokenizer(prompts, padding=True, return_tensors="pt")
        self.toks = tokenized_output.input_ids

        return self.toks

    def print_prompts(self, clean_prompts, corrupted_prompts):
        print("Clean Prompts:")
        for prompt in clean_prompts:
            print(prompt)
        print("\nCorrupted Prompts:")
        for prompt in corrupted_prompts:
            print(prompt)

    def get_dataset(self):
        # Clean prompts
        # clean_prompts = self.generate_clean_prompts()

        clean_prompts = [
            "Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in USA, Washington - Peter",
            "Lucy lives in Turkey, Ankara - Lucy, Sara lives in Italy, Rome - Sara, Bob lives in Spain, Madrid - Bob",
            "Tom lives in Canada, Toronto - Tom, Anna lives in Australia, Canberra - Anna, Michael lives in Japan, Tokyo - Michael",
            "David lives in Brazil, Rio de Janeiro - David, Alice lives in France, Paris - Alice, Peter lives in Germany, Berlin - Peter",
            "Sara lives in USA, Washington - Sara, Lucy lives in Turkey, Ankara - Lucy, Tom lives in Italy, Rome - Tom",
            "John lives in Spain, Madrid - John, Michael lives in Canada, Toronto - Michael, Anna lives in Australia, Canberra - Anna",
            "David lives in Japan, Tokyo - David, Sara lives in Brazil, Rio de Janeiro - Sara, Alice lives in France, Paris - Alice",
            "Bob lives in Germany, Berlin - Bob, Peter lives in USA, Washington - Peter, Lucy lives in Turkey, Ankara - Lucy",
            "Anna lives in Italy, Rome - Anna, Tom lives in Spain, Madrid - Tom, David lives in Canada, Toronto - David",
            "Michael lives in Australia, Canberra - Michael, John lives in Japan, Tokyo - John, Sara lives in Brazil, Rio de Janeiro - Sara",
            "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - John",
            "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - Bob",
            "Lucy lives in Spain, Madrid - Lucy, Michael lives in Canada, Toronto - Michael, Tom lives in Australia, Canberra - Tom",
            "Anna lives in Japan, Tokyo - Anna, Sara lives in Brazil, Rio de Janeiro - Sara, David lives in France, Paris - David",
            "John lives in Germany, Berlin - John, Peter lives in USA, Washington - Peter, Lucy lives in Turkey, Ankara - Lucy",
            "Tom lives in Italy, Rome - Tom, David lives in Spain, Madrid - David, Michael lives in Canada, Toronto - Michael",
            "Sara lives in Australia, Canberra - Sara, Alice lives in Japan, Tokyo - Alice, Bob lives in Brazil, Rio de Janeiro - Bob",
            "Peter lives in France, Paris - Peter, Lucy lives in Germany, Berlin - Lucy, Tom lives in USA, Washington - Tom",
            "David lives in Turkey, Ankara - David, Michael lives in Italy, Rome - Michael, Anna lives in Spain, Madrid - Anna",
            "John lives in Canada, Toronto - John, Sara lives in Australia, Canberra - Sara, Alice lives in Japan, Tokyo - Alice"
        ]
        
        # Counterfactual prompts
        corrupted_prompts = [
            "Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in Italy, Paris - Sara",
            "Lucy lives in Turkey, Ankara - Lucy, Sara lives in Italy, Rome - Sara, Bob lives in Canada, Berlin - Alice",
            "Tom lives in Canada, Toronto - Tom, Anna lives in Australia, Canberra - Anna, Michael lives in Brazil, Washington - Bob",
            "David lives in Brazil, Rio de Janeiro - David, Alice lives in France, Paris - Alice, Peter lives in Japan, Rome - Lucy",
            "Sara lives in USA, Washington - Sara, Lucy lives in Turkey, Ankara - Lucy, Tom lives in Spain, Madrid - Alice",
            "John lives in Spain, Madrid - John, Michael lives in Canada, Toronto - Michael, Anna lives in France, Berlin - Peter",
            "David lives in Japan, Tokyo - David, Sara lives in Brazil, Rio de Janeiro - Sara, Alice lives in Turkey, Madrid - Tom",
            "Bob lives in Germany, Berlin - Bob, Peter lives in USA, Washington - Peter, Lucy lives in Italy, Tokyo - Anna",
            "Anna lives in Italy, Rome - Anna, Tom lives in Spain, Madrid - Tom, David lives in Germany, Canberra - Michael",
            "Michael lives in Australia, Canberra - Michael, John lives in Japan, Tokyo - John, Sara lives in France, Berlin - David",
            "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in Spain, Madrid - Peter",
            "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in Canada, Tokyo - Lucy",
            "Lucy lives in Spain, Madrid - Lucy, Michael lives in Canada, Toronto - Michael, Tom lives in Japan, Washington - John",
            "Anna lives in Japan, Tokyo - Anna, Sara lives in Brazil, Rio de Janeiro - Sara, David lives in Germany, Berlin - Bob",
            "John lives in Germany, Berlin - John, Peter lives in USA, Washington - Peter, Lucy lives in Spain, Madrid - Michael",
            "Tom lives in Italy, Rome - Tom, David lives in Spain, Madrid - David, Michael lives in Japan, Canberra - John",
            "Sara lives in Australia, Canberra - Sara, Alice lives in Japan, Tokyo - Alice, Bob lives in Turkey, Paris - David",
            "Peter lives in France, Paris - Peter, Lucy lives in Germany, Berlin - Lucy, Tom lives in Canada, Rome - Sara",
            "David lives in Turkey, Ankara - David, Michael lives in Italy, Rome - Michael, Anna lives in Germany, Berlin - Bob",
            "John lives in Canada, Toronto - John, Sara lives in Australia, Canberra - Sara, Alice lives in Turkey, Washington - Peter"
        ]   
        # Print clean and corrupted prompts
        self.print_prompts(clean_prompts, corrupted_prompts)

        # Tokenize the prompts
        clean_prompts = self.tokenize_prompts(clean_prompts)
        corrupted_prompts = self.tokenize_prompts(corrupted_prompts)

        return clean_prompts, corrupted_prompts

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hybrid_retrieval_dataset = HybridRetrievalDataset()
    
    # Get datasets
    clean_data, patch_data = hybrid_retrieval_dataset.get_dataset()
    clean_data = clean_data.to(device)
    patch_data = patch_data.to(device)

    # # Print the tokenized datasets for verification
    # print("Clean Data Datasets:")
    # print(clean_data)

    # print("\nCorrupted Data Datasets:")
    # print(corrupted_data)

    # Define sequence length and number of examples
    # seq_len = cle
    # seq_len = 25
    # num_examples = 20

    # # Create the validation and test splits
    # default_data = clean_data[:num_examples*2, :seq_len - 1].to(device)
    # patch_data = patch_data[:num_examples*2, :seq_len - 1].to(device)
    # labels = clean_data[:num_examples*2, seq_len - 1].to(device)
    # wrong_labels = torch.as_tensor(clean_data[:num_examples*2, seq_len - 1], dtype=torch.long, device=device)

    # # Split into validation and test sets
    # validation_data = default_data[:num_examples, :]
    # validation_patch_data = patch_data[:num_examples, :]
    # validation_labels = labels[:num_examples]
    # validation_wrong_labels = wrong_labels[:num_examples]

    # test_data = default_data[num_examples:, :]
    # test_patch_data = patch_data[num_examples:, :]
    # test_labels = labels[num_examples:]
    # test_wrong_labels = wrong_labels[num_examples:]

    # # Print shapes for verification
    # print(f"Shape of validation_data: {validation_data.shape}")
    # print(f"Shape of validation_patch_data: {validation_patch_data.shape}")
    # print(f"Shape of validation_labels: {validation_labels.shape}")
    # print(f"Shape of validation_wrong_labels: {validation_wrong_labels.shape}")

    # print(f"Shape of test_data: {test_data.shape}")
    # print(f"Shape of test_patch_data: {test_patch_data.shape}")
    # print(f"Shape of test_labels: {test_labels.shape}")
    # print(f"Shape of test_wrong_labels: {test_wrong_labels.shape}")