import random
from typing import Tuple, List
from transformers import AutoTokenizer
import os
import torch
import copy
import re

# change root to ACDC parent directory
os.chdir("/home/iustin/Mech-Interp/Automatic-Circuit-Discovery")

NAMES = [
    "Alice",
    "John",
    "Peter", 
    "Bob", 
    "Tom", 
    "Anna", 
    "Michael", 
    "David",
    ]

COUNTRY_CAPITAL_PAIRS = [
    ('USA', 'Washington'), 
    ('Canada', 'Toronto'), 
    ('UK', 'London'), 
    ('Russia', 'Moscow'),
    ('France', 'Paris'),
    ]

TEMPLATE =  "{name1} lives in {country1}, {capital1} - {name1}, {name2} lives in {country2}, {capital2} - {name2}, {name3} lives in {country3}, {capital3} - {name3}"

def gen_prompts(names, template, country_capital_pairs, N, prefixes=None, seed=None):
    
    assert seed is not None
    random.seed(seed)

    nb_gen = 0 

    kbicr_prompts = []
    kbicr_corr_prompts = []
    
    while nb_gen < N:
        # Randomly select distinct names
        name1, name2, name3 = random.sample(names, 3)

        # Randomly select three distinct country-capital pairs
        (country1, capital1), (country2, capital2), (country3, capital3) = random.sample(country_capital_pairs, 3)
        
        # Replace the placeholders in the template with selected values
        prompt = template.format(name1=name1, country1=country1, capital1=capital1,
                                name2=name2, country2=country2, capital2=capital2,
                                name3=name3, country3=country3, capital3=capital3,
        )
        kbicr_prompt = {}
        if prefixes is not None:
            L = random.randint(30, 40)
            pref = ".".join(random.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""
        
        prompt = pref + template.replace("{name1}", name1).replace("{country1}", country1).replace("{capital1}", capital1) \
                    .replace("{name2}", name2).replace("{country2}", country2).replace("{capital2}", capital2) \
                    .replace("{name3}", name3).replace("{country3}", country3).replace("{capital3}", capital3)

        kbicr_prompt["text"] = prompt
        remove_last_name3 = lambda prompt: prompt[:prompt.rfind(name3)].rstrip(" ") if prompt.rfind(name3) != -1 else prompt
        kbicr_prompt["prompt"] = remove_last_name3(prompt)
        kbicr_prompt["S"] = name3
        kbicr_prompt["NON_S"] = name2
        kbicr_prompt["correct_label"] = " " + name3 
        kbicr_prompt["wrong_label"] = " " + name2 # or country_3
        kbicr_prompts.append(kbicr_prompt)

        # Corrupt the prompt by replacing name3 with name2 in all occurrences
        # \b ensures that only whole words are replaced
        corrupted_prompt = re.sub(rf"\b{name3}\b", "TEMP_NAME", prompt) # use temporary placeholder to avoid conflicts for all occurances of name3
        corrupted_prompt = re.sub(rf"\b{name2}\b", name3, corrupted_prompt)
        corrupted_prompt = re.sub(rf"\bTEMP_NAME\b", name2, corrupted_prompt)

        # Add prefixes to the corrupted prompt if they exist
        if prefixes is not None:
            corrupted_prompt = pref + corrupted_prompt

        kbicr_corr_prompt = {}
        kbicr_corr_prompt["text"] = corrupted_prompt
        remove_last_name2 = lambda corrupted_prompt: corrupted_prompt[:corrupted_prompt.rfind(name2)].rstrip(" ") if corrupted_prompt.rfind(name2) != -1 else corrupted_prompt
        kbicr_corr_prompt["prompt"] = remove_last_name2(corrupted_prompt)
        kbicr_corr_prompt["S"] = name2
        kbicr_corr_prompt["NON_S"] = name3
        kbicr_corr_prompt["correct_label"] = " "  + name2
        kbicr_corr_prompt["wrong_label"] = " " + name3
        kbicr_corr_prompts.append(kbicr_corr_prompt)

        nb_gen += 1
    
    return kbicr_prompts, kbicr_corr_prompts

class KBICRDataset:
    def __init__(self, prompts=None, corrupted_prompts=None, prefixes=None, N=20, tokenizer=None, prepend_bos=False, seed=None):
        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except OSError as e:
                print(f"Error loading tokenizer: {e}")
                print("Please ensure you have an active internet connection and the correct model identifier.")
        else:
            self.tokenizer = tokenizer

        self.prefixes = prefixes
        if prompts is None:
            self.kbicr_prompts, self.kbicr_corrupted_prompts = gen_prompts(NAMES, TEMPLATE, COUNTRY_CAPITAL_PAIRS, N=N, prefixes=self.prefixes, seed=seed)
        else:
            assert N == len(prompts), f"{N} and {len(prompts)}"
            self.kbicr_prompts = prompts
            self.kbicr_corrupted_prompts = corrupted_prompts

        print("Clean Prompts:")
        for prompt in self.kbicr_prompts:
            print(prompt["text"])
        print("\nCorrupted Prompts:")
        for corr_prompt in self.kbicr_corrupted_prompts:
            print(corr_prompt["text"])

        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.kbicr_prompts + self.kbicr_corrupted_prompts
        ]
        self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
            torch.int
        )
        self.N = N
        self.max_len = max(
            [
            len(self.tokenizer(prompt["text"]).input_ids)
            for prompt in self.kbicr_prompts + self.kbicr_corrupted_prompts
            ]
        )

        self.s_tokenIDs = [
            self.tokenizer.encode(" " + prompt["S"])[0] for prompt in self.kbicr_prompts
        ]
        self.non_s_tokenIDs = [
            self.tokenizer.encode(" " + prompt["NON_S"])[0] for prompt in self.kbicr_prompts
        ]
        self.s_token_decoded = [
            self.tokenizer.decode([self.tokenizer.encode(" " + prompt["S"])[0]])
            for prompt in self.kbicr_prompts
        ]
        self.non_s_token_decoded = [
            self.tokenizer.decode([self.tokenizer.encode(" " + prompt["NON_S"])[0]])
            for prompt in self.kbicr_prompts
        ]

        self.tokenized_prompts = []

        for i in range(self.N):
            self.tokenized_prompts.append(
            "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )

    def copy(self):
        copy_kbicr_dataset = KBICRDataset(
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=self.kbicr_prompts.copy(),
            prefixes=self.prefixes.copy()
            if self.prefixes is not None
            else self.prefixes,
        )
        return copy_kbicr_dataset

    def __getitem__(self, key):
        sliced_prompts = self.kbicr_prompts[key]
        sliced_dataset = KBICRDataset(
            N=len(sliced_prompts),
            tokenizer=self.tokenizer,
            prompts=sliced_prompts,
            prefixes=self.prefixes,
        )
        return sliced_dataset

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return self.N

    def tokenized_prompts(self):
        return self.toks

if __name__ == "__main__":
    dataset = KBICRDataset(N=3, seed=42)
        
    print(dataset.kbicr_corrupted_prompts)
    
    

    
