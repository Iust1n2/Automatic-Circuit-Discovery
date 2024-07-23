import sys
import os
os.chdir('/home/iustin/Mech-Interp/Automatic-Circuit-Discovery') # change to the root dir of the project

# # # Add the project root directory to the Python path
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
from acdc.hybridretrieval.datasets.kbicr_template_indirect import HybridRetrievalDataset  # NOTE: the only import that is different from the original code is for the task
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
    clean_data, corrupted_data, clean_labels, clean_wrong_labels = hybrid_retrieval_dataset.get_dataset()
    clean_data = clean_data.to(device)
    corrupted_data = corrupted_data.to(device)

    # Print the tensor for verification
    print("Clean Data Datasets:")
    print(clean_data.shape)

    print("\nCorrupted Data Datasets:")
    print(corrupted_data.shape)

    print("\nClean Labels shape:")
    print(clean_labels.shape)

    print("\nClean Wrong Labels shape:")
    print(clean_wrong_labels.shape)
    
    print("\nClean Labels shape:")
    print(clean_labels.shape)

    print("\nClean Wrong Labels shape:")
    print(clean_wrong_labels.shape)

    num_examples = clean_data.size(0)

    # Define the split
    validation_slice = num_examples // 2  # Half of the data for validation

    # Split the data
    validation_data = clean_data[:validation_slice, :]
    validation_labels = clean_labels[:validation_slice]
    validation_patch_data = corrupted_data[:validation_slice, :]
    validation_wrong_labels = clean_wrong_labels[:validation_slice]
    test_data = clean_data[validation_slice:, :]
    test_labels = clean_labels[validation_slice:]
    test_patch_data = corrupted_data[validation_slice:, :]
    test_wrong_labels = clean_wrong_labels[validation_slice:]

    # Checking the shapes of the resulting datasets
    print(f"Shape of validation_data: {validation_data.shape}")
    print(f"Shape of validation_patch_data: {validation_patch_data.shape}")
    print(f"Shape of test_data: {test_data.shape}")
    print(f"Shape of test_patch_data: {test_patch_data.shape}")

    with torch.no_grad():
        base_model_logits = tl_model(clean_data)[:, -1, :]
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

    base_validation_logprobs = base_model_logprobs[:validation_slice, :]
    base_test_logprobs = base_model_logprobs[validation_slice:, :]

    if metric_name == "logit_diff":
        validation_metric = partial(
                logit_diff_metric,
                correct_labels=validation_labels, # put none here instead of validation_labels
                wrong_labels=validation_wrong_labels, # put none here instead of validation_wrong_labels
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
            correct_labels=test_labels, # put none here instead of test_labels,
            wrong_labels=test_wrong_labels, # put none here instead of test_wrong_labels,
        )
    }
    
    return AllDataThings(
        tl_model=tl_model,
        validation_data=validation_data,
        validation_metric=validation_metric,
        validation_labels=validation_labels, 
        validation_wrong_labels=validation_wrong_labels, # put none here instead of validation_labels
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels, # put none here instead of test)labels
        test_wrong_labels=test_wrong_labels, 
        test_mask=None,
        test_patch_data=test_patch_data
    )

def main():
    num_examples = 20
    device = "cuda"
    metric_name = "kl_div"
    kl_return_one_element = True
    
    # Call the function to get all hybrid retrieval things
    all_data_things = get_all_hybrid_retrieval_things(num_examples, device, metric_name, kl_return_one_element)
    
    # Print the returned data
    print("All Data Things:")
    print(all_data_things)
    
    # Perform further operations or tests using the returned data
    
if __name__ == "__main__":
    main()
