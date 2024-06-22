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

def delete_substring_from_end(string: str, substring: str) -> str:
        if string.endswith(substring):
            return string[:len(string) - len(substring)]
        else:
            raise Exception(f"Substring {substring} is not contained at the end of the string {string}.")

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

        clean_prompts = [
            'Italy - Rome, Spain - Madrid, Canada - Ottawa, Spain - Madrid',
            'Japan - Tokyo, Australia - Canberra, Brazil - Brasília, France - Paris',
            'Turkey - Ankara, Russia - Moscow, India - New Delhi, China - Beijing',
            'China - Beijing, South Korea - Seoul, Mexico - Mexico City, Japan - Tokyo',
            'United Kingdom - London, Ireland - Dublin, New Zealand - Wellington, USA - Washington',
            'Sweden - Stockholm, Norway - Oslo, Finland - Helsinki, Denmark - Copenhagen',
            'Argentina - Buenos Aires, Chile - Santiago, Peru - Lima, Colombia - Bogotá',
            'South Africa - Pretoria, Egypt - Cairo, Nigeria - Abuja, Kenya - Nairobi',
            'Saudi Arabia - Riyadh, Romania - Bucharest, Qatar - Doha, Kuwait - Kuwait',
            'Greece - Athens, Portugal - Lisbon, Netherlands - Amsterdam, Belgium - Brussels',
            'Belgium - Brussels, Switzerland - Bern, Austria - Vienna, Luxembourg - Luxembourg',
            'Poland - Warsaw, Czech Republic - Prague, Hungary - Budapest, Slovakia - Bratislava',
            'Denmark - Copenhagen, Bulgaria - Sofia, Norway - Oslo, Finland - Helsinki',
            'Turkey - Istanbul, Lebanon - Beirut, Syria - Damascus, Jordan - Amman',
            'Thailand - Bangkok, Albania - Tirana, Malaysia - Kuala Lumpur, Singapore - Singapore',
            'Philippines - Manila, Indonesia - Jakarta, Singapore - Singapore, Romania - Bucharest',
            'Kenya - Nairobi, Sweden - Oslo, Uganda - Kampala, Rwanda - Kigali',
            'Colombia - Bogotá, Venezuela - Caracas, Ecuador - Quito, Peru - Lima',
            'Morocco - Rabat, Algeria - Algiers, Tunisia - Tunis, Libya - Tripoli',
            'Romania - Bucharest, Spain - Madrid, Canada - Ottawa, Germany - Berlin'
        ]

        clean_labels = clean_labels = [prompt.split(",")[-1].strip() for prompt in clean_prompts]
    
        corrupted_prompts = [
            'Italy - Rome, Spain - Brussels, Canada - Ottawa, Spain - Madrid',
            'Japan - Tokyo, Australia - Budapest, Brazil - Brasília, France - Paris',
            'Turkey - Ankara, Russia - Warsaw, India - New Delhi, China - Beijing',
            'China - Beijing, South Korea - Prague, Mexico - Mexico City, Japan - Tokyo',
            'United Kingdom - London, Ireland - Copenhagen, New Zealand - Wellington, USA - Washington',
            'Sweden - Stockholm, Norway - Paris, Finland - Helsinki, Denmark - Copenhagen',
            'Argentina - Buenos Aires, Chile - Vienna, Peru - Lima, Colombia - Bogotá',
            'South Africa - Pretoria, Egypt - Budapest, Nigeria - Abuja, Kenya - Nairobi',
            'Saudi Arabia - Riyadh, Romania - Berlin, Qatar - Doha, Kuwait - Kuwait',
            'Greece - Athens, Portugal - Sofia, Netherlands - Amsterdam, Belgium - Brussels',
            'Belgium - Brussels, Switzerland - Bucharest, Austria - Vienna, Luxembourg - Luxembourg',
            'Poland - Warsaw, Czech Republic - Madrid, Hungary - Budapest, Slovakia - Bratislava',
            'Denmark - Copenhagen, Bulgaria - Rome, Norway - Oslo, Finland - Helsinki',
            'Turkey - Ankara, Lebanon - Lima, Syria - Damascus, Jordan - Amman',
            'Thailand - Bangkok, Vietnam - Luxembourg, Malaysia - Kuala Lumpur, Singapore - Singapore',
            'Philippines - Manila, Indonesia - Hanoi, Singapore - Singapore, Romania - Bucharest',
            'Kenya - Nairobi, Sweden - Paris, Uganda - Kampala, Rwanda - Kigali',
            'Colombia - Bogotá, Venezuela - New Delhi, Ecuador - Quito, Peru - Lima',
            'Morocco - Rabat, Algeria - Luxembourg, Tunisia - Tunis, Libya - Tripoli',
            'Romania - Bucharest, Spain - Rome, Canada - Ottawa, Germany - Berlin'
        ]

        corrupted_labels = clean_labels = [prompt.split(",")[-1].strip() for prompt in corrupted_prompts]

        # Print clean and corrupted prompts
        self.print_prompts(clean_prompts, corrupted_prompts)

        clean_prompts = list(map(delete_substring_from_end, clean_prompts, clean_labels))
        corrupted_prompts = list(map(delete_substring_from_end, corrupted_prompts, corrupted_labels))

        # Tokenize the prompts
        clean_prompts = self.tokenize_prompts(clean_prompts)
        corrupted_prompts = self.tokenize_prompts(corrupted_prompts)
        clean_labels = self.tokenize_prompts(clean_labels)
        corrupted_labels = self.tokenize_prompts(corrupted_labels)
        
        return clean_prompts, corrupted_prompts

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hybrid_retrieval_dataset = HybridRetrievalDataset()
    
    # Get datasets
    clean_data, patch_data = hybrid_retrieval_dataset.get_dataset()
    clean_data = clean_data.to(device)
    patch_data = patch_data.to(device)

    # Print the tokenized datasets for verification
    print("Clean Data Datasets:")
    print(clean_data.shape)

    print("\nCorrupted Data Datasets:")
    print(patch_data.shape)