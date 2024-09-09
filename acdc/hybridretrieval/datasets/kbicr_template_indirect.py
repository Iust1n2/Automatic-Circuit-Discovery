# Majority of the code is taken from the ioi_dataset.py script, 
# only the gen_prompts and corrupt_prompts functions differ because the task is inherently different.

import random
from typing import Tuple, List
from transformers import AutoTokenizer
import os
import torch
import copy
import re
import warnings
import numpy as np

# change path to root directory to load gpt-2 stuff 
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

    prompts = []
    
    while nb_gen < N:
        temp = template
        temp_id = template.index(temp)
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
        kbicr_prompt["TEMPLATE_IDX"] = temp_id
        prompts.append(kbicr_prompt)

        nb_gen += 1
    
    return prompts

def corrupt_prompts(prompts, seed=None):
    
    assert seed is not None
    random.seed(seed)
    kbicr_corr_prompts = []

    for prompt in prompts:
        corrupted_prompt = re.sub(rf"\b{prompt['S']}\b", "TEMP_NAME", prompt["text"])  # Swap 'S' with temporary placeholder
        corrupted_prompt = re.sub(rf"\b{prompt['NON_S']}\b", prompt['S'], corrupted_prompt)  # Swap 'NON_S' with 'S'
        corrupted_prompt = re.sub(rf"\bTEMP_NAME\b", prompt['NON_S'], corrupted_prompt)  # Restore 'NON_S' in place of the placeholder
        
        new_prompt = prompt.copy()  # Copy the original prompt dictionary
        new_prompt["text"] = corrupted_prompt  
        remove_last_name2 = lambda text, name: text[:text.rfind(name)].rstrip(" ") if text.rfind(name) != -1 else text
        new_prompt["prompt"] = remove_last_name2(corrupted_prompt, prompt['NON_S'])
        new_prompt["S"] = prompt['NON_S']
        new_prompt["NON_S"] = prompt['S']
        new_prompt["correct_label"] = " " + prompt['NON_S']
        new_prompt["wrong_label"] = " " + prompt['S']
        kbicr_corr_prompts.append(new_prompt)

    return kbicr_corr_prompts

# *Tok Idxs Methods

# {name1} lives in {country1}, {capital1} - {name1}, {NON_S1} lives in {country2}, {capital2} - {NON_S2}, {S1} lives in {country3}, {capital3} - {S2}

def get_name_idxs(prompts, tokenizer, idx_types=["NON_S", "S", "S2"], prepend_bos=False):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)
    double_s2 = False
    for prompt in prompts:
        t = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(t[:-1]))
        for idx_type in idx_types:
            if "2" in idx_type:
                idx = (
                    len(toks)
                    - toks[::-1].index(
                        tokenizer.tokenize(" " + prompt[idx_type[:-1]])[0]
                    )
                    - 1
                )
            else:
                idx = toks.index(tokenizer.tokenize(" " + prompt[idx_type])[0])
            name_idx_dict[idx_type].append(idx)
        if "S" in idx_types and "S2" in idx_types:
            if name_idx_dict["S"][-1] == name_idx_dict["S2"][-1]:
                double_s2 = True
    if double_s2:
        warnings.warn("S2 index has been computed as the same for S and S2")

    return [
        int(prepend_bos) + torch.tensor(name_idx_dict[idx_type])
        for idx_type in idx_types
    ]

def get_word_idxs(prompts, word_list, tokenizer):
    """Get the index of the words in word_list in the prompts. Exactly one of the word_list word has to be present in each prompt"""
    idxs = []
    tokenized_words = [
        tokenizer.decode(tokenizer(word)["input_ids"][0]) for word in word_list
    ]
    for pr_idx, prompt in enumerate(prompts):
        toks = [
            tokenizer.decode(t)
            for t in tokenizer(prompt["text"], return_tensors="pt", padding=True)[
                "input_ids"
            ][0]
        ]
        idx = None
        for i, w_tok in enumerate(tokenized_words):
            if word_list[i] in prompt["text"]:
                try:
                    idx = toks.index(w_tok)
                    if toks.count(w_tok) > 1:
                        idx = len(toks) - toks[::-1].index(w_tok) - 1
                except:
                    idx = toks.index(w_tok)
                    raise ValueError(toks, w_tok, prompt["text"])
        if idx is None:
            raise ValueError(f"Word {word_list} and {i} not found {prompt}")
        idxs.append(idx)
    return torch.tensor(idxs)


def get_end_idxs(prompts, tokenizer, name_tok_len=1, prepend_bos=False, toks=None):
    # toks = torch.Tensor(tokenizer([prompt["text"] for prompt in prompts], padding=True).input_ids).type(torch.int)
    relevant_idx = int(prepend_bos)
    # if the sentence begins with an end token
    # AND the model pads at the end with the same end token,
    # then we need make special arrangements

    pad_token_id = tokenizer.pad_token_id

    end_idxs_raw = []
    for i in range(toks.shape[0]):
        if pad_token_id not in toks[i][1:]:
            end_idxs_raw.append(toks.shape[1])
            continue
        nonzers = (toks[i] == pad_token_id).nonzero()
        try:
            nonzers = nonzers[relevant_idx]
        except:
            print(toks[i])
            print(nonzers)
            print(relevant_idx)
            print(i)
            raise ValueError("Something went wrong")
        nonzers = nonzers[0]
        nonzers = nonzers.item()
        end_idxs_raw.append(nonzers)
    end_idxs = torch.tensor(end_idxs_raw)
    end_idxs = end_idxs - 1 - name_tok_len

    for i in range(toks.shape[0]):
        assert toks[i][end_idxs[i] + 1] != 0 and (
            toks.shape[1] == end_idxs[i] + 2 or toks[i][end_idxs[i] + 2] == pad_token_id
        ), (
            toks[i],
            end_idxs[i],
            toks[i].shape,
            "the END idxs aren't properly formatted",
        )

    return end_idxs

ALL_SEM = [
    "NON_S",
    "S",
    "S2",
    "-",
    "S+1",
    "end",
]


def get_idx_dict(prompts, tokenizer, prepend_bos=False, toks=None):
    (NON_S_idxs, S1_idxs, S2_idxs,) = get_name_idxs(
        prompts,
        tokenizer,
        idx_types=["NON_S", "S", "S2"],
        prepend_bos=prepend_bos,
    )

    end_idxs = get_end_idxs(
        prompts,
        tokenizer,
        name_tok_len=1,
        prepend_bos=prepend_bos,
        toks=toks,
    )

    punct_idxs = get_word_idxs(prompts, [",", "."], tokenizer)
    dash_idxs = get_word_idxs(prompts, [" -"], tokenizer)
    # TODO: check if indices are correctly indexed 
    return {
        "NON_S": NON_S_idxs,
        "NON_S-1": NON_S_idxs - 1,
        "NON_S+1": NON_S_idxs + 1,
        "S": S1_idxs,
        "S-1": S1_idxs - 1,
        "S+1": S1_idxs + 1,
        "S2": S2_idxs,
        "end": end_idxs,
        "-": dash_idxs,
        "starts": torch.zeros_like(end_idxs),
        "punct": punct_idxs,
    }

class KBICRDataset:
    def __init__(
        self, 
        prompts=None, 
        prefixes=None, 
        N=20, 
        tokenizer=None, 
        kbicr_prompts_for_word_idxs=None,
        prepend_bos=False, 
        manual_word_idx=None,
        seed=None
        ):
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
            self.prompts = gen_prompts(NAMES, TEMPLATE, COUNTRY_CAPITAL_PAIRS, N=N, prefixes=self.prefixes, seed=seed)
        else:
            assert N == len(prompts), f"{N} and {len(prompts)}"
            self.prompts = prompts

        # print("Clean Prompts:")
        # for prompt in self.prompts:
        #     print(prompt["text"])

        self.sentences = [prompt["text"] for prompt in self.prompts]

        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.prompts
        ]
        self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
            torch.int
        )

        if kbicr_prompts_for_word_idxs is None:
            kbicr_prompts_for_word_idxs = self.prompts

        self.word_idx = get_idx_dict(
            kbicr_prompts_for_word_idxs,
            self.tokenizer,
            prepend_bos=prepend_bos,
            toks=self.toks,
        )
        self.prepend_bos = prepend_bos
        if manual_word_idx is not None:
            self.word_idx = manual_word_idx

        self.sem_tok_idx = {
            k: v for k, v in self.word_idx.items() if k in ALL_SEM
        }

        self.N = N
        self.max_len = max(
            [
            len(self.tokenizer(prompt["text"]).input_ids)
            for prompt in self.prompts
            ]
        )

        self.s_tokenIDs = [
            self.tokenizer.encode(" " + prompt["S"])[0] for prompt in self.prompts
        ]
        self.non_s_tokenIDs = [
            self.tokenizer.encode(" " + prompt["NON_S"])[0] for prompt in self.prompts
        ]
        self.s_token_decoded = [
            self.tokenizer.decode([self.tokenizer.encode(" " + prompt["S"])[0]])
            for prompt in self.prompts
        ]
        self.non_s_token_decoded = [
            self.tokenizer.decode([self.tokenizer.encode(" " + prompt["NON_S"])[0]])
            for prompt in self.prompts
        ]
        # for each template, get the indices of the prompts that use that template
        # self.groups is used for mean ablations in HookedTransformer2.ipynb in the Circuit Validation part
        all_ids = [prompt["TEMPLATE_IDX"] for prompt in self.prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        self.tokenized_prompts = []

        for i in range(self.N):
            self.tokenized_prompts.append(
            "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )
            
    def gen_corrupted_dataset(self, seed=None):

        assert seed is not None
        random.seed(seed)
        
        corr_prompts = corrupt_prompts(self.prompts, seed=seed)

        corr_dataset = KBICRDataset(
            prompts=corr_prompts,
            N=self.N,
            tokenizer=self.tokenizer,
            prefixes=self.prefixes,
            seed=seed,
        ) 
        return corr_dataset

    def copy(self):
        copy_kbicr_dataset = KBICRDataset(
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=self.prompts.copy(),
            prefixes=self.prefixes.copy()
            if self.prefixes is not None
            else self.prefixes,
        )
        return copy_kbicr_dataset

    def __getitem__(self, key):
        sliced_prompts = self.prompts[key]
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

# testing
if __name__ == "__main__":
    dataset = KBICRDataset(N=1, seed=42)
    corr_dataset = dataset.gen_corrupted_dataset(seed=42)

    print(dataset.prompts)
    # print(corr_dataset.prompts)
    print(dataset.groups)    