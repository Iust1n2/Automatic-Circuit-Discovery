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
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
    
    def tokenize_prompt(self, prompt: str, prepend_bos=False):
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            prompt = str(prompt)
        # Tokenize the prompt
        
        texts = (self.tokenizer.bos_token if prepend_bos else "") + prompt
        
        tokenized_output = self.tokenizer(texts, padding=True, return_tensors="pt")
        self.toks = tokenized_output.input_ids
        self.tokenized_prompts = self.toks

        return self.tokenized_prompts

    def get_dataset(self, factual=True):
        # Factual prompts
        factual_prompts = [
            "alice lives in france,\nparis - alice,\nbob lives in germany,\nberlin - bob,\njohn lives in usa,\nwashington - john ",
            "alice lives france,\nparis - alice,\nbob lives germany,\nberlin - bob,\njohn lives USA,\nwashington - john",
            "alice resides france,\nparis - alice,\nbob resides germany,\nberlin - bob,\njohn resides usa,\nwashington - john",
            "alice resides in france,\nparis - alice,\nbob resides in germany,\nberlin - bob,\njohn resides in usa,\nwashington - john",
            "alice works france,\nparis - alice,\nbob works germany,\nberlin - bob,\njohn works usa,\nwashington - john",
            "alice loves france,\nparis - alice,\nbob loves germany,\nberlin - bob,\njohn loves usa,\nwashington - john",
            "alice travels france,\nparis - alice,\nbob travels germany,\nberlin - bob,\njohn travels usa,\nwashington - john",
            "alice is French,\nparis - alice,\nbob is German,\nberlin - bob,\njohn is american,\nwashington - john",
            "alice is from france,\nparis - alice,\nbob is from germany,\nberlin - bob,\njohn is from usa,\nwashington - john"
        ]
        
        # Counterfactual prompts
        counterfactual_prompts = [
            "alice drinks france,\nparis - alice,\nbob drinks germany,\nberlin - bob,\njohn drinks usa,\nwashington - john",
            "alice cuts france,\nparis - alice,\nbob cuts germany,\nberlin - bob,\njohn cuts usa,\nwashington - john",
            "alice drills france,\nparis - alice,\nbob drills germany,\nberlin - bob,\njohn drills usa,\nwashington - john"
        ]

        # Choose the appropriate group of prompts
        prompts = factual_prompts if factual else counterfactual_prompts
        # return [self.load_single_example(prompt) for prompt in prompts]
        tokenized_prompt = self.tokenize_prompt(prompts)
        # tokenized_prompt = self.tokenize_prompt(prompt)
        return tokenized_prompt

# Usage
if __name__ == "__main__":
    hybrid_retrieval_dataset = HybridRetrievalDataset()
    
    # Get factual prompts
    factual_datasets = hybrid_retrieval_dataset.get_dataset(factual=True)
    print("Factual Prompts Datasets:")
    for dataset in factual_datasets:
        print(dataset)
    
    # Get counterfactual prompts
    counterfactual_datasets = hybrid_retrieval_dataset.get_dataset(factual=False)
    print("\nCounterfactual Prompts Datasets:")
    for dataset in counterfactual_datasets:
        print(dataset)
    seq_len = 29
    # input = factual_datasets.long().flatten()[:seq_len-1]
    # target = factual_datasets.long().flatten()[seq_len-1]
    # print(f"Input: {input.shape}, target: {target.shape}")
    input = factual_datasets[:, :seq_len-1]
    target = factual_datasets[:, seq_len-1]
    patch_input = counterfactual_datasets[:, :seq_len-1]
    patch_target = counterfactual_datasets[:, seq_len-1]

    print(f"Shape of input: {input.shape}")
    print(f"Shape of target: {target.shape}")
    print(f"Shape of patch_input: {patch_input.shape}")
    print(f"Shape of patch_target: {patch_target.shape}")
   