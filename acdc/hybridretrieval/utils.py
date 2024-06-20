from collections import OrderedDict
from dataclasses import dataclass
from acdc.TLACDCEdge import (
    Edge,
    EdgeType,
    TorchIndex,
)
from acdc.acdc_utils import filter_nodes, get_present_nodes
from acdc.TLACDCInterpNode import TLACDCInterpNode
import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
import IPython
from acdc.acdc_utils import MatchNLLMetric, frac_correct_metric, logit_diff_metric, kl_divergence, negative_log_probs
import torch
from acdc.docstring.utils import AllDataThings
from acdc.hybridretrieval.hybrid_retrieval_dataset4 import HybridRetrievalDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm
import wandb
from transformer_lens.HookedTransformer import HookedTransformer

def get_gpt2_small(device="cuda") -> HookedTransformer:
    tl_model = HookedTransformer.from_pretrained("gpt2")
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model
def get_hybrid_retrieval_gpt2_small(device="cuda"):
    """For backwards compat"""
    return get_gpt2_small(device=device)

def get_model(device):
    tl_model = HookedTransformer.from_pretrained(
        "gpt2-small",  # load gpt-2 small
        center_writing_weights=False,  # these are needed as this model is a Shortformer; this is a technical detail
        center_unembed=False,
        fold_ln=False,
        device=device,
    )

    # standard ACDC options
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True) 
    return tl_model

def get_all_hybrid_retrieval_things(num_examples, device, metric_name, kl_return_one_element=True):
    tl_model = get_gpt2_small(device=device)
    hybrid_retrieval_dataset = HybridRetrievalDataset()
    
    # Get datasets
    clean_data, corrupted_data = hybrid_retrieval_dataset.get_dataset()
    clean_data = clean_data.to(device)
    corrupted_data = corrupted_data.to(device)

    # Print the tokenized datasets for verification
    print("Clean Data Datasets:")
    print(clean_data)

    print("\nCorrupted Data Datasets:")
    print(corrupted_data)

    # Define sequence length and number of examples
    seq_len = 25
    # seq_len = clean_data.shape[1]
    assert seq_len == 25, f"Well, I thought Hybrid-Retrieval was 16 not {seq_len} tokens long..."

    # Create the validation and test splits
    default_data = clean_data[:num_examples*2, :seq_len - 1].to(device)
    patch_data = corrupted_data[:num_examples*2, :seq_len - 1].to(device)
    labels = clean_data[:num_examples*2, seq_len - 1].to(device)
    wrong_labels = torch.as_tensor(corrupted_data[:num_examples*2, seq_len - 1], dtype=torch.long, device=device)

    # Split into validation and test sets
    validation_data = default_data[:num_examples, :]
    validation_patch_data = patch_data[:num_examples, :]
    validation_labels = labels[:num_examples]
    validation_wrong_labels = wrong_labels[:num_examples]

    test_data = default_data[:num_examples, :]
    test_patch_data = patch_data[:num_examples, :]
    test_labels = labels[:num_examples]
    test_wrong_labels = wrong_labels[:num_examples]

    # Print shapes for verification
    print(f"Shape of validation_data: {validation_data.shape}")
    print(f"Shape of validation_patch_data: {validation_patch_data.shape}")
    print(f"Shape of validation_labels: {validation_labels.shape}")
    print(f"Shape of validation_wrong_labels: {validation_wrong_labels.shape}")

    print(f"Shape of test_data: {test_data.shape}")
    print(f"Shape of test_patch_data: {test_patch_data.shape}")
    print(f"Shape of test_labels: {test_labels.shape}")
    print(f"Shape of test_wrong_labels: {test_wrong_labels.shape}")

    with torch.no_grad():
        base_model_logits = tl_model(default_data)[:, -1, :]
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

    base_validation_logprobs = base_model_logprobs[:num_examples, :]
    base_test_logprobs = base_model_logprobs[num_examples:, :]

    if metric_name == "logit_diff":
        validation_metric = partial(
                logit_diff_metric,
                correct_labels=validation_labels,
                wrong_labels=validation_wrong_labels,
            )
    elif metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=kl_return_one_element,
        )
    else:
        raise ValueError(f"metric_name {metric_name} not recognized")
    
    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
        ),
        "logit_diff": partial(
            logit_diff_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        )
    }
    

    
    return AllDataThings(
        tl_model=tl_model,
        validation_data=validation_data,
        validation_metric=validation_metric,
        validation_labels=validation_labels,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data
    )
