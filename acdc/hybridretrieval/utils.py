import sys
import os
os.chdir('/home/iustin/Mech-Interp/Automatic-Circuit-Discovery') # change to the root dir of the project

# Add the project root directory to the Python path
sys.path.insert(0, os.getcwd())

import dataclasses
from typing import Optional, Callable, Any
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
from acdc.hybridretrieval.datasets.kbicr_template_indirect import KBICRDataset
from tqdm import tqdm
import wandb
from transformer_lens.HookedTransformer import HookedTransformer

@dataclasses.dataclass(frozen=False)
class AllDataThings:
    tl_model: HookedTransformer
    validation_metric: Callable[[torch.Tensor], torch.Tensor]
    validation_data: torch.Tensor
    validation_labels: Optional[torch.Tensor]
    validation_wrong_labels: Optional[torch.Tensor]
    validation_mask: Optional[torch.Tensor]
    validation_patch_data: torch.Tensor
    test_metrics: dict[str, Any]
    test_data: torch.Tensor
    test_labels: Optional[torch.Tensor]
    test_wrong_labels: Optional[torch.Tensor]
    test_mask: Optional[torch.Tensor]
    test_patch_data: torch.Tensor

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
        center_writing_weights=False,
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

    clean_dataset = KBICRDataset(N=num_examples*2, seed=42)
    corr_dataset = clean_dataset.gen_corrupted_dataset(seed=42)

    seq_len = clean_dataset.toks.shape[1]

    clean_prompts = clean_dataset.toks.long()[:num_examples*2, :seq_len - 1].to(device)
    corrupted_prompts = corr_dataset.toks.long()[:num_examples*2, :seq_len - 1].to(device) 
    clean_prompts_correct_answers = clean_dataset.toks.long()[:num_examples*2, seq_len - 1]
    clean_prompts_wrong_answers = torch.as_tensor(clean_dataset.non_s_tokenIDs[:num_examples*2], dtype=torch.long)

    assert torch.equal(clean_prompts_correct_answers, torch.as_tensor(clean_dataset.s_tokenIDs, dtype=torch.long))
    assert torch.equal(clean_prompts_wrong_answers, torch.as_tensor(clean_dataset.non_s_tokenIDs, dtype=torch.long))
    
    clean_prompts_correct_answers = clean_prompts_correct_answers.to(device)
    clean_prompts_wrong_answers = clean_prompts_wrong_answers.to(device)
    
    # Print the tensor for verification
    print(f"\nClean Data Shape: {clean_prompts.shape}")
    print(f"\nCorrupted Data Shape: {corrupted_prompts.shape}")
    print(f"\nClean Labels shape: {clean_prompts_correct_answers.shape}")
    print(f"\nClean Wrong Labels shape:{clean_prompts_wrong_answers.shape}")

    # validation_slice = num_examples // 2  

    # Split the data
    validation_data = clean_prompts[:num_examples, :]
    validation_labels = clean_prompts_correct_answers[:num_examples]
    validation_patch_data = corrupted_prompts[:num_examples, :]
    validation_wrong_labels = clean_prompts_wrong_answers[:num_examples]
    test_data = clean_prompts[num_examples:, :]
    test_labels = clean_prompts_correct_answers[num_examples:]
    test_patch_data = corrupted_prompts[num_examples:, :]
    test_wrong_labels = clean_prompts_wrong_answers[num_examples:]

    # Checking the shapes of the resulting datasets
    print(f"\nShape of validation_data: {validation_data.shape}")
    print(f"\nShape of validation_patch_data: {validation_patch_data.shape}")
    print(f"\nShape of test_data: {test_data.shape}")
    print(f"\nShape of test_patch_data: {test_patch_data.shape}\n")

    with torch.no_grad():
        base_model_logits = tl_model(clean_prompts)[:, -1, :]
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1) # log_softmax results in negative values, because the logarithm of a no. < 1 is negative

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
        validation_wrong_labels=validation_wrong_labels, 
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels, 
        test_wrong_labels=test_wrong_labels, 
        test_mask=None,
        test_patch_data=test_patch_data
    )
# testing
def main():
    num_examples = 20
    device = "cuda"
    metric_name = "kl_div"
    kl_return_one_element = True
    
    all_data_things = get_all_hybrid_retrieval_things(num_examples, device, metric_name, kl_return_one_element)
    
    print("All Data Things:")
    print(all_data_things)
    
if __name__ == "__main__":
    main()
