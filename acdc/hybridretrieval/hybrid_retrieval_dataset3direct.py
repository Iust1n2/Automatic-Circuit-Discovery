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
            "Alice lives in France, Alice - France, Bob lives in Germany, Bob - Germany, John lives in USA, John - USA",
            "Lucy lives in Italy, Lucy - Italy, Tom lives in Spain, Tom - Spain, Sara lives in Canada, Sara - Canada",
            "Michael lives in Japan, Michael - Japan, Anna lives in Australia, Anna - Australia, David lives in Brazil, David - Brazil",
            "Peter lives in Turkey, Peter - Turkey, Alice lives in Russia, Alice - Russia, Bob lives in China, Bob - China",
            "John lives in India, John - India, Sara lives in South Korea, Sara - South Korea, Tom lives in Mexico, Tom - Mexico",
            "Lucy lives in United Kingdom, Lucy - United Kingdom, David lives in Ireland, David - Ireland, Anna lives in Norway, Anna - Norway",
            "Michael lives in Sweden, Michael - Sweden, Peter lives in Norway, Peter - Norway, Alice lives in Finland, Alice - Finland",
            "Bob lives in Argentina, Bob - Argentina, John lives in Chile, John - Chile, Sara lives in Peru, Sara - Peru",
            "Tom lives in South Africa, Tom - South Africa, Lucy lives in Egypt, Lucy - Egypt, David lives in Nigeria, David - Nigeria",
            "Anna lives in Saudi Arabia, Anna - Saudi Arabia, Michael lives in UAE, Michael - UAE, Peter lives in Qatar, Peter - Qatar",
            "Alice lives in Greece, Alice - Greece, Bob lives in Portugal, Bob - Portugal, John lives in Netherlands, John - Netherlands",
            "Sara lives in Belgium, Sara - Belgium, Tom lives in Switzerland, Tom - Switzerland, Lucy lives in Austria, Lucy - Austria",
            "David lives in Poland, David - Poland, Anna lives in Czech Republic, Anna - Czech Republic, Michael lives in Hungary, Michael - Hungary",
            "Peter lives in Denmark, Peter - Denmark, Alice lives in Iceland, Alice - Iceland, Bob lives in Norway, Bob - Norway",
            "John lives in Thailand, John - Thailand, Sara lives in Vietnam, Sara - Vietnam, Tom lives in Malaysia, Tom - Malaysia",
            "Lucy lives in Philippines, Lucy - Philippines, David lives in Indonesia, David - Indonesia, Anna lives in Singapore, Anna - Singapore",
            "Michael lives in Kenya, Michael - Kenya, Peter lives in Tanzania, Peter - Tanzania, Alice lives in Uganda, Alice - Uganda",
            "Bob lives in Colombia, Bob - Colombia, John lives in Venezuela, John - Venezuela, Sara lives in Ecuador, Sara - Ecuador",
            "Tom lives in Morocco, Tom - Morocco, Lucy lives in Algeria, Lucy - Algeria, David lives in Tunisia, David - Tunisia",
            "Anna lives in Jordan, Anna - Jordan, Michael lives in Lebanon, Michael - Lebanon, Peter lives in Syria, Peter - Syria"
                    ]

        clean_labels = [prompt.split(",")[-1].strip() for prompt in clean_prompts]

        corrupted_prompts = [
            "Alice lives in France, Alice - France, Bob lives in Germany, Bob - Germany, John lives in USA, Alice - France",
            "Lucy lives in Italy, Lucy - Italy, Tom lives in Spain, Tom - Spain, Sara lives in Canada, Michael - Canada",
            "Michael lives in Japan, Michael - Japan, Anna lives in Australia, Anna - Australia, David lives in Brazil, Anna - Australia",
            "Peter lives in Turkey, Peter - Turkey, Alice lives in Russia, Alice - Russia, Bob lives in China, Damian - China",
            "John lives in India, John - India, Sara lives in South Korea, Sara - South Korea, Tom lives in Mexico, John - India",
            "Lucy lives in United Kingdom, Lucy - United Kingdom, David lives in Ireland, David - Ireland, Anna lives in Bulgaria, Mary - Bulgaria",
            "Michael lives in Sweden, Michael - Sweden, Peter lives in Norway, Peter - Norway, Alice lives in Finland, Peter - Norway",
            "Bob lives in Argentina, Bob - Argentina, John lives in Chile, John - Chile, Sara lives in Peru, Justin - Peru",
            "Tom lives in South Africa, Tom - South Africa, Lucy lives in Egypt, Lucy - Egypt, David lives in Nigeria, Lucy - Egypt",
            "Anna lives in Saudi Arabia, Anna - Saudi Arabia, Michael lives in UAE, Michael - UAE, Peter lives in Qatar, David - Qatar",
            "Alice lives in Greece, Alice - Greece, Bob lives in Portugal, Bob - Portugal, John lives in Netherlands, Bob - Portugal",
            "Sara lives in Belgium, Sara - Belgium, Tom lives in Switzerland, Tom - Switzerland, Lucy lives in Austria, Sara - Belgium",
            "David lives in Poland, David - Poland, Anna lives in Czech Republic, Anna - Czech Republic, Michael lives in Hungary, Cesar - Hungary",
            "Peter lives in Denmark, Peter - Denmark, Alice lives in Iceland, Alice - Iceland, Bob lives in Norway, Alice - Iceland",
            "John lives in Thailand, John - Thailand, Sara lives in Vietnam, Sara - Vietnam, Tom lives in Malaysia, Peter - Malaysia",
            "Lucy lives in Philippines, Lucy - Philippines, David lives in Indonesia, David - Indonesia, Anna lives in Singapore, David - Indonesia",
            "Michael lives in Kenya, Michael - Kenya, Peter lives in Tanzania, Peter - Tanzania, Alice lives in Uganda, Peter - Tanzania",
            "Bob lives in Colombia, Bob - Colombia, John lives in Venezuela, John - Venezuela, Sara lives in Ecuador, Aubrey - Ecuador",
            "Tom lives in Morocco, Tom - Morocco, Lucy lives in Algeria, Lucy - Algeria, David lives in Tunisia, Tom - Morocco",
            "Anna lives in Jordan, Anna - Jordan, Michael lives in Lebanon, Michael - Lebanon, Peter lives in Syria, Dorian - Syria"
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