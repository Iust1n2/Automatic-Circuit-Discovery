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
            "Alice lives in France, Alice - Paris, John lives in Germany, John - Berlin, Peter lives in USA, Peter - Washington",
            "Lucy lives in Turkey, Lucy - Ankara, Sara lives in Italy, Sara - Rome, Bob lives in Spain, Bob - Madrid",
            "Tom lives in Canada, Tom - Toronto, Anna lives in Australia, Anna - Canberra, Michael lives in Japan, Michael - Tokyo",
            "David lives in Brazil, David - Rio de Janeiro, Alice lives in France, Alice - Paris, Peter lives in Germany, Peter - Berlin",
            "Sara lives in USA, Sara - Washington, Lucy lives in Turkey, Lucy - Ankara, Tom lives in Italy, Tom - Rome",
            "John lives in Spain, John - Madrid, Michael lives in Canada, Michael - Toronto, Anna lives in Australia, Anna - Canberra",
            "David lives in Japan, David - Tokyo, Sara lives in Brazil, Sara - Rio de Janeiro, Alice lives in France, Alice - Paris",
            "Bob lives in Germany, Bob - Berlin, Peter lives in USA, Peter - Washington, Lucy lives in Turkey, Lucy - Ankara",
            "Anna lives in Italy, Anna - Rome, Tom lives in Spain, Tom - Madrid, David lives in Canada, David - Toronto",
            "Michael lives in Australia, Michael - Canberra, John lives in Japan, John - Tokyo, Sara lives in Brazil, Sara - Rio de Janeiro",
            "Alice lives in France, Alice - Paris, Bob lives in Germany, Bob - Berlin, John lives in USA, John - Washington",
            "Peter lives in Turkey, Peter - Ankara, Alice lives in Italy, Alice - Rome, Bob lives in France, Bob - Paris",
            "Lucy lives in Spain, Lucy - Madrid, Michael lives in Canada, Michael - Toronto, Tom lives in Australia, Tom - Canberra",
            "Anna lives in Japan, Anna - Tokyo, Sara lives in Brazil, Sara - Rio de Janeiro, David lives in France, David - Paris",
            "John lives in Germany, John - Berlin, Peter lives in USA, Peter - Washington, Lucy lives in Turkey, Lucy - Ankara",
            "Tom lives in Italy, Tom - Rome, David lives in Spain, David - Madrid, Michael lives in Canada, Michael - Toronto",
            "Sara lives in Australia, Sara - Canberra, Alice lives in Japan, Alice - Tokyo, Bob lives in Brazil, Bob - Rio de Janeiro",
            "Peter lives in France, Peter - Paris, Lucy lives in Germany, Lucy - Berlin, Tom lives in USA, Tom - Washington",
            "David lives in Turkey, David - Ankara, Michael lives in Italy, Michael - Rome, Anna lives in Spain, Anna - Madrid",
            "John lives in Canada, John - Toronto, Sara lives in Australia, Sara - Canberra, Alice lives in Japan, Alice - Tokyo"
        ]

            
        clean_labels = [prompt.split(",")[-1].strip() for prompt in clean_prompts]

        corrupted_prompts = [
            "Alice lives in France, Alice - Paris, John lives in Germany, John - Berlin, Peter lives in USA, Michael - Washington",
            "Lucy lives in Turkey, Lucy - Ankara, Sara lives in Italy, Sara - Rome, Bob lives in Spain, Tom - Madrid",
            "Tom lives in Canada, Tom - Toronto, Anna lives in Australia, Anna - Canberra, Michael lives in Japan, David - Tokyo",
            "David lives in Brazil, David - Rio de Janeiro, Alice lives in France, Alice - Paris, Peter lives in Germany, Sara - Berlin",
            "Sara lives in USA, Sara - Washington, Lucy lives in Turkey, Lucy - Ankara, Tom lives in Italy, John - Rome",
            "John lives in Spain, John - Madrid, Michael lives in Canada, Michael - Toronto, Anna lives in Australia, Peter - Canberra",
            "David lives in Japan, David - Tokyo, Sara lives in Brazil, Sara - Rio de Janeiro, Alice lives in France, Lucy - Paris",
            "Bob lives in Germany, Bob - Berlin, Peter lives in USA, Peter - Washington, Lucy lives in Turkey, Anna - Ankara",
            "Anna lives in Italy, Anna - Rome, Tom lives in Spain, Tom - Madrid, David lives in Canada, Michael - Toronto",
            "Michael lives in Australia, Michael - Canberra, John lives in Japan, John - Tokyo, Sara lives in Brazil, Alice - Rio de Janeiro",
            "Alice lives in France, Alice - Paris, Bob lives in Germany, Bob - Berlin, John lives in USA, Tom - Washington",
            "Peter lives in Turkey, Peter - Ankara, Alice lives in Italy, Alice - Rome, Bob lives in France, David - Paris",
            "Lucy lives in Spain, Lucy - Madrid, Michael lives in Canada, Michael - Toronto, Tom lives in Australia, Sara - Canberra",
            "Anna lives in Japan, Anna - Tokyo, Sara lives in Brazil, Sara - Rio de Janeiro, David lives in France, John - Paris",
            "John lives in Germany, John - Berlin, Peter lives in USA, Peter - Washington, Lucy lives in Turkey, Michael - Ankara",
            "Tom lives in Italy, Tom - Rome, David lives in Spain, David - Madrid, Michael lives in Canada, Alice - Toronto",
            "Sara lives in Australia, Sara - Canberra, Alice lives in Japan, Alice - Tokyo, Bob lives in Brazil, Anna - Rio de Janeiro",
            "Peter lives in France, Peter - Paris, Lucy lives in Germany, Lucy - Berlin, Tom lives in USA, John - Washington",
            "David lives in Turkey, David - Ankara, Michael lives in Italy, Michael - Rome, Anna lives in Spain, Sara - Madrid",
            "John lives in Canada, John - Toronto, Sara lives in Australia, Sara - Canberra, Alice lives in Japan, Bob - Tokyo"
        ]

        corrupted_labels = [prompt.split(",")[-1].strip() for prompt in corrupted_prompts]

        ## old corrupted prompts
        # corrupted_prompts = [
        #     "Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in Italy, Canada - Peter",
        #     "Lucy lives in Turkey, Ankara - Lucy, Sara lives in Italy, Rome - Sara, Bob lives in Canada, Berlin - Alice",
        #     "Tom lives in Canada, Toronto - Tom, Anna lives in Australia, Canberra - Anna, Michael lives in Brazil, Washington - Bob",
        #     "David lives in Brazil, Rio de Janeiro - David, Alice lives in France, Paris - Alice, Peter lives in Japan, Rome - Lucy",
        #     "Sara lives in USA, Washington - Sara, Lucy lives in Turkey, Ankara - Lucy, Tom lives in Spain, Madrid - Alice",
        #     "John lives in Spain, Madrid - John, Michael lives in Canada, Toronto - Michael, Anna lives in France, Berlin - Peter",
        #     "David lives in Japan, Tokyo - David, Sara lives in Brazil, Rio de Janeiro - Sara, Alice lives in Turkey, Madrid - Tom",
        #     "Bob lives in Germany, Berlin - Bob, Peter lives in USA, Washington - Peter, Lucy lives in Italy, Tokyo - Anna",
        #     "Anna lives in Italy, Rome - Anna, Tom lives in Spain, Madrid - Tom, David lives in Germany, Canberra - Michael",
        #     "Michael lives in Australia, Canberra - Michael, John lives in Japan, Tokyo - John, Sara lives in France, Berlin - David",
        #     "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in Spain, Madrid - Peter",
        #     "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in Canada, Tokyo - Lucy",
        #     "Lucy lives in Spain, Madrid - Lucy, Michael lives in Canada, Toronto - Michael, Tom lives in Japan, Washington - John",
        #     "Anna lives in Japan, Tokyo - Anna, Sara lives in Brazil, Rio de Janeiro - Sara, David lives in Germany, Berlin - Bob",
        #     "John lives in Germany, Berlin - John, Peter lives in USA, Washington - Peter, Lucy lives in Spain, Madrid - Michael",
        #     "Tom lives in Italy, Rome - Tom, David lives in Spain, Madrid - David, Michael lives in Japan, Canberra - John",
        #     "Sara lives in Australia, Canberra - Sara, Alice lives in Japan, Tokyo - Alice, Bob lives in Turkey, Paris - David",
        #     "Peter lives in France, Paris - Peter, Lucy lives in Germany, Berlin - Lucy, Tom lives in Canada, Rome - Sara",
        #     "David lives in Turkey, Ankara - David, Michael lives in Italy, Rome - Michael, Anna lives in Germany, Berlin - Bob",
        #     "John lives in Canada, Toronto - John, Sara lives in Australia, Canberra - Sara, Alice lives in Turkey, Washington - Peter"
        # ]   

        
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