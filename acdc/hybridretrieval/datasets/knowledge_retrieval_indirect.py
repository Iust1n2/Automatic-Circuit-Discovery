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
            'Rome - Italy, Madrid - Spain, Ottawa - Canada, Berlin - Germany',
            'Tokyo - Japan, Canberra - Australia, Brasília - Brazil, Paris - France',
            'Ankara - Turkey, Moscow - Russia, New Delhi - India, Beijing - China',
            'Beijing - China, Seoul - South Korea, Mexico City - Mexico, Tokyo - Japan',
            'London - United Kingdom, Dublin - Ireland, Wellington - New Zealand, Washington - USA',
            'Stockholm - Sweden, Oslo - Norway, Helsinki - Finland, Copenhagen - Denmark',
            'Buenos Aires - Argentina, Santiago - Chile, Lima - Peru, Bogotá - Colombia',
            'Pretoria - South Africa, Cairo - Egypt, Abuja - Nigeria, Nairobi - Kenya',
            'Riyadh - Saudi Arabia, Abu Dhabi - UAE, Doha - Qatar, Kuwait City - Kuwait',
            'Athens - Greece, Lisbon - Portugal, Amsterdam - Netherlands, Brussels - Belgium',
            'Brussels - Belgium, Bern - Switzerland, Vienna - Austria, Luxembourg - Luxembourg',
            'Warsaw - Poland, Prague - Czech Republic, Budapest - Hungary, Bratislava - Slovakia',
            'Copenhagen - Denmark, Athens - Greece, Oslo - Norway, Helsinki - Finland',
            'Istanbul - Turkey, Beirut - Lebanon, Damascus - Syria, Amman - Jordan',
            'Bangkok - Thailand, Hanoi - Vietnam, Kuala Lumpur - Malaysia, Singapore - Singapore',
            'Manila - Philippines, Jakarta - Indonesia, Singapore - Singapore, Madrid - Spain',
            'Nairobi - Kenya, Dodoma - Tanzania, Kampala - Uganda, Kigali - Rwanda',
            'Bogotá - Colombia, Caracas - Venezuela, Quito - Ecuador, Lima - Peru',
            'Rabat - Morocco, Algiers - Algeria, Tunis - Tunisia, Tripoli - Libya',
            'Bucharest - Romania, Madrid - Spain, Ottawa - Canada, Berlin - Germany'
        ]

        clean_labels = [prompt.split(",")[-1].strip() for prompt in clean_prompts]


        corrupted_prompts = [
            'Rome - Italy, Paris - Spain, Ottawa - Canada, Berlin - Germany',
            'Tokyo - Japan, Budapest - Australia, Brasília - Brazil, Paris - France',
            'Ankara - Turkey, Warsaw - Russia, New Delhi - India, Beijing - China',
            'Beijing - China, Oslo - South Korea, Mexico City - Mexico, Tokyo - Japan',
            'London - United Kingdom, Copenhagen - Ireland, Wellington - New Zealand, Washington - USA',
            'Stockholm - Sweden, Rome - Norway, Helsinki - Finland, Copenhagen - Denmark',
            'Buenos Aires - Argentina, Vienna - Chile, Lima - Peru, Bogotá - Colombia',
            'Pretoria - South Africa, Madrid - Egypt, Abuja - Nigeria, Nairobi - Kenya',
            'Riyadh - Saudi Arabia, Cape Town - UAE, Doha - Qatar, Kuwait City - Kuwait',
            'Athens - Greece, Tokyo - Portugal, Amsterdam - Netherlands, Brussels - Belgium',
            'Brussels - Belgium, Chisinau - Switzerland, Vienna - Austria, Luxembourg - Luxembourg',
            'Warsaw - Poland, Madrid - Czech Republic, Budapest - Hungary, Bratislava - Slovakia',
            'Copenhagen - Denmark, Ottawa - Greece, Oslo - Norway, Helsinki - Finland',
            'Istanbul - Turkey, Guatemala City - Lebanon, Damascus - Syria, Amman - Jordan',
            'Bangkok - Thailand, Luxembourg - Vietnam, Kuala Lumpur - Malaysia, Singapore - Singapore',
            'Manila - Philippines, Madrid - Indonesia, Singapore - Singapore, Madrid - Spain',
            'Nairobi - Kenya, Oslo - Tanzania, Kampala - Uganda, Kigali - Rwanda',
            'Bogotá - Colombia, Ankara - Venezuela, Quito - Ecuador, Lima - Peru',
            'Rabat - Morocco, Vienna - Algeria, Tunis - Tunisia, Tripoli - Libya',
            'Bucharest - Romania, Bratislava - Spain, Ottawa - Canada, Berlin - Germany'
            ]

        corrupted_labels = [prompt.split(",")[-1].strip() for prompt in corrupted_prompts]

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