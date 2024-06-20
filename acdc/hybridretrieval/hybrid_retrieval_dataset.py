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

class HybridRetrievalDataset:
    def __init__(self, prompt, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

    # def load_single_example(self, prompt: str):
    #     # Split the prompt into relevant parts
    #     lines = prompt.split('\n')
    #     data = {
    #         "input": [],
    #         "target": []
    #     }
        
    #     for line in lines:
    #         if '-' in line:
    #             parts = line.split('-')
    #             location = parts[0].strip()
    #             name = parts[1].strip()
    #             if "Washington" in location:
    #                 data["input"].append(location)
    #                 data["target"].append(name)
    #             else:
    #                 data["input"].append(line.strip())
        
    #     return data
    
    def tokenize_prompt(self, prompt, prepend_bos=False):
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            prompt = str(prompt)
        # Tokenize the prompt
        
        texts = (self.tokenizer.bos_token if prepend_bos else "") + prompt
        
        self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
            torch.int)
        
        return self.toks
    
    def get_dataset(self, prompt: str):
        # Load the single example
        # prompt="Alice lives France,\nParis - Alice,\nBob lives Germany,\nBerlin - Bob,\nJohn lives USA,\nWashington - John"
        tokenized_prompt = self.tokenize_prompt(prompt)
        # tokenized_prompt = self.tokenize_prompt(prompt)
        return tokenized_prompt

# Usage
if __name__ == "__main__":
    prompt = "Alice lives France,\nParis - Alice,\nBob lives Germany,\nBerlin - Bob,\nJohn lives USA,\nWashington - John"
    single_example_dataset = HybridRetrievalDataset(prompt)
    single_example_dataset = single_example_dataset.tokenize_prompt(prompt)
    seq_len = 29
    input = single_example_dataset.long()[:seq_len-1]
    target = single_example_dataset.long()[seq_len-1]
    print(input, target)

    # print(single_example_dataset["input"])
    # print(single_example_dataset["target"])