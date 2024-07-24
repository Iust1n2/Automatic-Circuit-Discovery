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
import copy
from itertools import cycle, islice

names = ["Alice", "John", "Peter", "Bob", "Tom", "Anna", "Michael", "David", "Lucy", "Sara"]
capitals = ["Paris", "Berlin", "Washington", "Ankara", "Rome", "Madrid", "Toronto", "London", "Tokyo", "Lima"]
countries = ["France", "Germany", "USA", "Turkey", "Italy", "Spain", "Canada", "UK", "Japan", "Chile"]
# for corrupting the prompts
capitals_out = ["Cairo", "Vienna", "Oslo", "Warsaw"]
countries_out = ["Egypt", "Austria", "Finland", "Poland"]

class HybridRetrievalDataset:
    def __init__(self, prompts=None, tokenizer=None):
        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except OSError as e:
                print(f"Error loading tokenizer: {e}")
                print("Please ensure you have an active internet connection and the correct model identifier.")
        else:
            self.tokenizer = tokenizer

    def tokenize_prompts(self, prompts: list):
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

    def get_dataset(self, clean_prompts, corrupted_prompts, clean_labels, corrupted_labels):
        self.print_prompts(clean_prompts, corrupted_prompts)

        # Tokenize the prompts
        clean_prompts = self.tokenize_prompts(clean_prompts)
        corrupted_prompts = self.tokenize_prompts(corrupted_prompts)
        clean_labels = self.tokenize_prompts(clean_labels)
        corrupted_labels = self.tokenize_prompts(corrupted_labels)
        
        return clean_prompts, corrupted_prompts

    def generate_prompts_and_labels(self, names, capitals, countries):
        template = "{name1} lives in {country1}, {capital1} - {name1}, {name2} lives in {country2}, {capital2} - {name2}, {name3} lives in {country3}, {capital3} - "

        clean_prompts = []
        clean_labels = []
        wrong_labels = []

        indices = list(range(len(names)))
        for idx1, idx2, idx3 in zip(cycle(indices), cycle(indices[1:]), cycle(indices[2:])):
            prompt = template.format(
                name1=names[idx1], country1=countries[idx1], capital1=capitals[idx1],
                name2=names[idx2], country2=countries[idx2], capital2=capitals[idx2],
                name3=names[idx3], country3=countries[idx3], capital3=capitals[idx3]
            )
            label = names[idx3]
            wrong_label = countries[idx3]
            clean_prompts.append(prompt)
            clean_labels.append(label)
            wrong_labels.append(wrong_label)
            
            if len(clean_prompts) >= 20:
                break

        return clean_prompts, clean_labels, wrong_labels

    def generate_corrupted_prompts_and_labels(self, names, capitals, countries):
        # first 2 templates are for capital 3 and country 3 out of context
        template1 = "{name1} lives in {country1}, {capital1} - {name1}, {name2} lives in {country2}, {capital2} - {name2}, {name3} lives in {country3}, {capital4} - "
        template2 = "{name1} lives in {country1}, {capital1} - {name1}, {name2} lives in {country2}, {capital2} - {name2}, {name3} lives in {country4}, {capital3} - "
        # last 2 templates are for capital 3 context
        template3 = "{name1} lives in {country1}, {capital1} - {name1}, {name2} lives in {country2}, {capital2} - {name2}, {name3} lives in {country3}, {capital1} - "
        template4 = "{name1} lives in {country1}, {capital1} - {name1}, {name2} lives in {country2}, {capital2} - {name2}, {name3} lives in {country3}, {capital2} - "
        corrupted_prompts = []
        corrupted_labels = []

        indices = list(range(len(names)))
        for idx1, idx2, idx3, idx4 in zip(cycle(indices), cycle(indices[1:]), cycle(indices[2:]), cycle(indices[:3])):
            if len(corrupted_prompts) >= 20:
                break

            if len(corrupted_prompts) % 3 == 0:
                prompt = template1.format(
                    name1=names[idx1], country1=countries[idx1], capital1=capitals[idx1],
                    name2=names[idx2], country2=countries[idx2], capital2=capitals[idx2],
                    name3=names[idx3], country3=countries[idx3], capital4=capitals_out[idx4]
                )
            elif len(corrupted_prompts) % 3 == 1:
                prompt = template2.format(
                    name1=names[idx1], country1=countries[idx1], capital1=capitals[idx1],
                    name2=names[idx2], country2=countries[idx2], capital2=capitals[idx2],
                    name3=names[idx3], country4=countries_out[idx4], capital3=capitals[idx3]
                )
            elif len(corrupted_prompts) % 4 == 2:
                prompt = template3.format(
                    name1=names[idx1], country1=countries[idx1], capital1=capitals[idx1],
                    name2=names[idx2], country2=countries[idx2], capital2=capitals[idx2],
                    name3=names[idx3], country3=countries[idx3], capital3=capitals[idx1]
                )
            else:
                prompt = template4.format(
                    name1=names[idx1], country1=countries[idx1], capital1=capitals[idx1],
                    name2=names[idx2], country2=countries[idx2], capital2=capitals[idx2],
                    name3=names[idx3], country3=countries[idx3], capital3=capitals[idx2]
                )

            label = names[idx3]
            corrupted_prompts.append(prompt)
            corrupted_labels.append(label)

        return corrupted_prompts, corrupted_labels
                 
    def get_dataset(self):
        clean_prompts, clean_labels, clean_wrong_labels = self.generate_prompts_and_labels(names, capitals, countries)
        corrupted_prompts, corrupted_labels = self.generate_corrupted_prompts_and_labels(names, capitals, countries)
        
        self.print_prompts(clean_prompts, corrupted_prompts)

        # Tokenize the prompts
        clean_prompts = self.tokenize_prompts(clean_prompts)
        corrupted_prompts = self.tokenize_prompts(corrupted_prompts)
        clean_labels = self.tokenize_prompts(clean_labels)
        corrupted_labels = self.tokenize_prompts(corrupted_labels)
        clean_wrong_labels = self.tokenize_prompts(clean_wrong_labels)

        return clean_prompts, corrupted_prompts, clean_labels, clean_wrong_labels
    
if __name__ == "__main__":
    dataset = HybridRetrievalDataset()
    clean_prompts, clean_labels, wrong_labels = dataset.generate_prompts_and_labels(names, capitals, countries)
    corrupted_prompts, corrupted_labels = dataset.generate_corrupted_prompts_and_labels(names, capitals, countries)
    print("Generated Prompts:\n")
    for prompt in clean_prompts:
        print(prompt)
    # print("\nGenerated Labels:")
    # for label in clean_labels:
    #     print(label)
    # print("\nGenerated Wrong Labels:")
    # for wrong_label in wrong_labels:
    #     print(wrong_label)
    print("\nCorrupted Prompts:")
    for prompt in corrupted_prompts:
        print(prompt)
    # print("\nCorrupted Labels:")
    # for label in corrupted_labels:
    #     print(label)
    
    # for prompt, label, wrong_label in zip(clean_prompts, clean_labels, wrong_labels):
    #     print(f"Prompt: {prompt}\nLabel: {label}\nWrong Label: {wrong_label}\n")
    # for prompt, label in zip(corrupted_prompts, corrupted_labels):
    #     print(f"Corrupted Prompt: {prompt}\nLabel: {label}\n")