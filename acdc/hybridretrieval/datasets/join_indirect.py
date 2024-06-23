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
        # Clean prompts
        # clean_prompts = self.generate_clean_prompts()

        clean_prompts = [
            'Alice lives in France, France - Alice, Bob lives in Germany, Germany - Bob, John lives in USA, USA - John',
            'Lucy lives in Italy, Italy - Lucy, Tom lives in Spain, Spain - Tom, Sara lives in Canada, Canada - Sara',
            'Michael lives in Japan, Japan - Michael, Anna lives in Australia, Australia - Anna, David lives in Brazil, Brazil - David',
            'Peter lives in Turkey, Turkey - Peter, Alice lives in Russia, Russia - Alice, Bob lives in China, China - Bob',
            'John lives in India, India - John, Sara lives in South Korea, South Korea - Sara, Tom lives in Mexico, Mexico - Tom',
            'Lucy lives in United Kingdom, United Kingdom - Lucy, David lives in Ireland, Ireland - David, Anna lives in New Zealand, New Zealand - Anna',
            'Michael lives in Sweden, Sweden - Michael, Peter lives in Norway, Norway - Peter, Alice lives in Finland, Finland - Alice',
            'Bob lives in Argentina, Argentina - Bob, John lives in Chile, Chile - John, Sara lives in Peru, Peru - Sara',
            'Tom lives in South Africa, South Africa - Tom, Lucy lives in Egypt, Egypt - Lucy, David lives in Nigeria, Nigeria - David',
            'Anna lives in Saudi Arabia, Saudi Arabia - Anna, Michael lives in UAE, UAE - Michael, Peter lives in Qatar, Qatar - Peter',
            'Alice lives in Greece, Greece - Alice, Bob lives in Portugal, Portugal - Bob, John lives in Netherlands, Netherlands - John',
            'Sara lives in Belgium, Belgium - Sara, Tom lives in Switzerland, Switzerland - Tom, Lucy lives in Austria, Austria - Lucy',
            'David lives in Poland, Poland - David, Anna lives in Czech Republic, Czech Republic - Anna, Michael lives in Hungary, Hungary - Michael',
            'Peter lives in Denmark, Denmark - Peter, Alice lives in Iceland, Iceland - Alice, Bob lives in Norway, Norway - Bob',
            'John lives in Thailand, Thailand - John, Sara lives in Vietnam, Vietnam - Sara, Tom lives in Malaysia, Malaysia - Tom',
            'Lucy lives in Philippines, Philippines - Lucy, David lives in Indonesia, Indonesia - David, Anna lives in Singapore, Singapore - Anna',
            'Michael lives in Kenya, Kenya - Michael, Peter lives in Tanzania, Tanzania - Peter, Alice lives in Uganda, Uganda - Alice',
            'Bob lives in Colombia, Colombia - Bob, John lives in Venezuela, Venezuela - John, Sara lives in Ecuador, Ecuador - Sara',
            'Tom lives in Morocco, Morocco - Tom, Lucy lives in Algeria, Algeria - Lucy, David lives in Tunisia, Tunisia - David',
            'Anna lives in Jordan, Jordan - Anna, Michael lives in Lebanon, Lebanon - Michael, Peter lives in Syria, Syria - Peter',
        ]

        clean_labels = [prompt.split(",")[-1].strip() for prompt in clean_prompts]

    
        corrupted_prompts = [

            "Alice lives in France, France - Alice, Bob lives in Germany, Germany - Bob, John lives in USA, Peru - John",
            "Lucy lives in Italy, Italy - Lucy, Tom lives in Spain, Spain - Tom, Sara lives in Canada, Italy - Sara",
            "Michael lives in Japan, Japan - Michael, Anna lives in Australia, Australia - Anna, David lives in Brazil, Bulgaria - David",
            "Peter lives in Turkey, Turkey - Peter, Alice lives in Russia, Russia - Alice, Bob lives in China, Turkey - Bob",            
            "John lives in India, India - John, Sara lives in South Korea, South Korea - Sara, Tom lives in Mexico, Germany - Tom",
            "Lucy lives in United Kingdom, United Kingdom - Lucy, David lives in Ireland, Ireland - David, Anna lives in New Zealand, Ireland - Anna",
            "Michael lives in Sweden, Sweden - Michael, Peter lives in Norway, Norway - Peter, Alice lives in Finland, Estonia - Alice",
            "Bob lives in Argentina, Argentina - Bob, John lives in Chile, Chile - John, Sara lives in Peru, Chile - Sara",
            "Tom lives in South Africa, South Africa - Tom, Lucy lives in Egypt, Egypt - Lucy, David lives in Nigeria, Zimbabwe - David",
            "Anna lives in Saudi Arabia, Saudi Arabia - Anna, Michael lives in UAE, UAE - Michael, Peter lives in Qatar, Saudi Arabia - Peter",
            "Alice lives in Greece, Greece - Alice, Bob lives in Portugal, Portugal - Bob, John lives in Netherlands, Belgium - John",
            "Sara lives in Belgium, Belgium - Sara, Tom lives in Switzerland, Switzerland - Tom, Lucy lives in Austria, Belgium - Lucy",
            "David lives in Poland, Poland - David, Anna lives in Czech Republic, Czech Republic - Anna, Michael lives in Hungary, Slovakia - Michael",
            "Peter lives in Denmark, Denmark - Peter, Alice lives in Iceland, Iceland - Alice, Bob lives in Norway, Iceland - Bob",
            "John lives in Thailand, Thailand - John, Sara lives in Vietnam, Vietnam - Sara, Tom lives in Malaysia, Indonesia - Tom",
            "Lucy lives in Philippines, Philippines - Lucy, David lives in Indonesia, Indonesia - David, Anna lives in Singapore, Brazil - Anna",
            "Michael lives in Kenya, Kenya - Michael, Peter lives in Tanzania, Tanzania - Peter, Alice lives in Uganda, Tanzania - Alice",
            "Bob lives in Colombia, Colombia - Bob, John lives in Venezuela, Venezuela - John, Sara lives in Ecuador, Mexico - Sara",
            "Tom lives in Morocco, Morocco - Tom, Lucy lives in Algeria, Algeria - Lucy, David lives in Tunisia, Morocco - David",
            "Anna lives in Jordan, Jordan - Anna, Michael lives in Lebanon, Lebanon - Michael, Peter lives in Syria, Moldova - Peter",
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
