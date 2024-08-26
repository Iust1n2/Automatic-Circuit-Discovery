import random
from typing import Tuple, List
from transformers import AutoTokenizer
import pathlib
import os
import sys
os.getcwd()

# change root to ACDC parent directory
os.chdir("/home/iustin/Mech-Interp/Automatic-Circuit-Discovery")

names = [
    "Alice",
    "John",
    "Peter", 
    "Bob", 
    "Tom", 
    "Anna", 
    "Michael", 
    "David"
    ]

country_capital_pairs = [
    ('USA', 'Washington'), 
    ('Canada', 'Toronto'), 
    ('UK', 'London'), 
    ('Russia', 'Moscow'),
    ('France', 'Paris'),
    ]

country_capital_pairs_out = [ 
    ('France', 'Paris')
    ]

capitals_out = [
    'Paris', 
    'Brazil'
    ]


def gen_prompts(names: Tuple[str], 
                country_capital_pairs: Tuple[List[str]], 
                N: int, 
                seed: int=None):
    
    assert seed is not None
    random.seed(seed)

    generated_prompts = []
    correct_labels = []
    wrong_labels = []

    template =  "{name1} lives in {country1}, {capital1} - {name1}, {name2} lives in {country2}, {capital2} - {name2}, {name3} lives in {country3}, {capital3} - "
    
    for _ in range(N):        
        # Randomly select distinct names
        name1, name2, name3 = random.sample(names, 3)

        # Randomly select three distinct country-capital pairs
        (country1, capital1), (country2, capital2), (country3, capital3) = random.sample(country_capital_pairs, 3)
        
        # Replace the placeholders in the template with selected values
        prompt = template.format(name1=name1, country1=country1, capital1=capital1,
                                name2=name2, country2=country2, capital2=capital2,
                                name3=name3, country3=country3, capital3=capital3,
        )
        
        correct_labels.append(name3)
        wrong_labels.append(country3)
        generated_prompts.append(prompt)
    
    return generated_prompts, correct_labels, wrong_labels

def corrupt_prompts(clean_prompts: Tuple[str], 
                    seed: int=None):
    
    if seed is not None:
        random.seed(seed)

    # Corruption is replacing the name3 with the name2 
    corruption_template = "{name1} lives in {country1}, {capital1} - {name1}, {name3} lives in {country2}, {capital2} - {name3}, {name2} lives in {country3}, {capital3} - "
    
    corrupted_prompts = []
    corrupted_labels = []
    corrupted_wrong_labels = [] 
    
    for clean_prompt in clean_prompts:

        parts = clean_prompt.split(" ")
        if len(parts) < 19:
            raise ValueError("The clean prompt format is not valid.")

        name1 = parts[0]
        country1 = parts[3].split(",")[0]
        capital1 = parts[4]
        name2 = parts[7]
        country2 = parts[10].split(",")[0]
        capital2 = parts[11]
        name3 = parts[14]
        country3 = parts[17].split(",")[0]
        capital3 = parts[18]
        
        # Apply the corruption
        corrupted_prompt = corruption_template.format(
            name1 = name1, country1 = country1, capital1 = capital1,
            name3 = name3, country2 = country2, capital2 = capital2,
            name2 = name2, country3 = country3, capital3 = capital3
        )

        corrupted_prompts.append(corrupted_prompt)
        corrupted_labels.append(name2)
        corrupted_wrong_labels.append(name3)

    return corrupted_prompts, corrupted_labels, corrupted_wrong_labels

class KBICRDataset:
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except OSError as e:
                print(f"Error loading tokenizer: {e}")
                print("Please ensure you have an active internet connection and the correct model identifier.")
        else:
            self.tokenizer = tokenizer

    def tokenize_prompts(self, prompts: Tuple[str]):
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
        clean_prompts, clean_labels, clean_wrong_labels = gen_prompts(names, country_capital_pairs, N=20, seed=42)
        corrupted_prompts, corrupted_labels, corrupted_wrong_labels = corrupt_prompts(clean_prompts, seed=42)
        
        self.print_prompts(clean_prompts, corrupted_prompts)

        # Tokenize the prompts
        clean_prompts = self.tokenize_prompts(clean_prompts)
        corrupted_prompts = self.tokenize_prompts(corrupted_prompts)
        clean_labels = self.tokenize_prompts(clean_labels)
        corrupted_labels = self.tokenize_prompts(corrupted_labels)
        clean_wrong_labels = self.tokenize_prompts(clean_wrong_labels)
        corrupted_wrong_labels = self.tokenize_prompts(corrupted_wrong_labels)

        return clean_prompts, clean_labels, clean_wrong_labels, corrupted_prompts, corrupted_labels, corrupted_wrong_labels

if __name__ == "__main__":
    dataset = KBICRDataset()
    clean_prompts, clean_labels, clean_wrong_labels, corrupted_prompts, corrupted_labels, corrupted_wrong_labels = dataset.get_dataset()
    
    # for prompt, label, wrong_label in zip(clean_prompts, clean_labels, clean_wrong_labels):
    #     print(f"Prompt: {prompt}\nLabel: {label}\nWrong Label: {wrong_label}\n")
    # for prompt, label, wrong_label in zip(corrupted_prompts, corrupted_labels, corrupted_wrong_labels):
    #     print(f"Corrupted Prompt: {prompt}\nLabel: {label}\nWrong Label: {wrong_label}\n")

    # Decode the tokenized prompts, labels, and wrong labels
    decoded_clean_prompts = [dataset.tokenizer.decode(prompt) for prompt in clean_prompts]
    decoded_clean_labels = [dataset.tokenizer.decode(label) for label in clean_labels]
    decoded_clean_wrong_labels = [dataset.tokenizer.decode(wrong_label) for wrong_label in clean_wrong_labels]

    decoded_corrupted_prompts = [dataset.tokenizer.decode(prompt) for prompt in corrupted_prompts]
    decoded_corrupted_labels = [dataset.tokenizer.decode(label) for label in corrupted_labels]
    decoded_corrupted_wrong_labels = [dataset.tokenizer.decode(wrong_label) for wrong_label in corrupted_wrong_labels]

    # # Print the decoded prompts, labels, and wrong labels
    # print("Decoded Clean Prompts:")
    # for prompt, label, wrong_label in zip(decoded_clean_prompts, decoded_clean_labels, decoded_clean_wrong_labels):
    #     print(f"Clean Prompt: {prompt}\nLabel: {label}\nWrong Label: {wrong_label}\n")

    # print("Decoded Corrupted Prompts:")
    # for prompt, label, wrong_label in zip(decoded_corrupted_prompts, decoded_corrupted_labels, decoded_corrupted_wrong_labels):
    #     print(f"Corrupted Prompt: {prompt}\nLabel: {label}\nWrong Label: {wrong_label}\n")

    # Print decoded clean labels
    print("Decoded Clean Labels:")
    for label in decoded_clean_labels:
        print(label)
    
    # Print decoded corrupted labels
    print("\nDecoded Corrupted Labels:")
    for label in decoded_corrupted_labels:
        print(label)
    
