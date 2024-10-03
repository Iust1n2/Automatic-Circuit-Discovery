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
from acdc.hybridretrieval.datasets.kbicr_template_indirect import KBICRDataset, TEMPLATE1
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

def get_all_hybrid_retrieval_things(num_examples, device, metric_name, kl_return_one_element=True):
    tl_model = get_gpt2_small(device=device)

    clean_dataset = KBICRDataset(N=num_examples*2, template=TEMPLATE1, seed=42)
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

# first iteration
# CIRCUIT = { 
#     "name mover": [(9, 9), (8, 11), (9, 6), (10, 0)],
#     "backup name mover": [(10, 6), (10, 10)],
#     "negative name mover": [(10, 7), (11, 10), (11, 6)],
#     "country-capital mover": [(9, 8)],
#     "induction": [(5, 5), (5, 1), (6, 9), (7, 10), (7, 2)],
#     "duplicate token": [(0, 5), (3, 0), (0, 1)],
#     "previous token": [(4, 11), (3, 7), (6, 8), (2, 2)],
# }

CIRCUIT = {
    "name mover": [(9, 9), (8, 11), (9, 6), (10, 0)],
    "backup name mover": [(10, 6), (10, 10), (11, 2), (11, 1), (11, 6)],
    "negative name mover": [(10, 7), (11, 10), (11, 6)],
    "country-capital mover": [(9, 8)],
    "non-s inhibition": [(8, 10), (7, 7)],
    "induction": [(5, 5), (5, 1), (7, 1), (7, 10), (6, 9)],
    "duplicate token": [(0, 10), (1, 11), (0, 5), (3, 0), (0, 1)],
    "previous token": [(2, 2), (4, 11)],
}

@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: tuple[str, ...]

def get_kbicr_true_edges(model):
    all_groups_of_nodes = [group for _, group in CIRCUIT.items()]
    all_nodes = [node for group in all_groups_of_nodes for node in group]
    assert len(all_nodes) == 27, len(all_nodes)

    nodes_to_mask = []

    for layer_idx in range(12):
        for head_idx in range(12):
            if (layer_idx, head_idx) not in all_nodes:
                for letter in ["q", "k", "v"]:
                    nodes_to_mask.append(
                        TLACDCInterpNode(name=f"blocks.{layer_idx}.attn.hook_{letter}", index = TorchIndex([None, None, head_idx]), incoming_edge_type=EdgeType.DIRECT_COMPUTATION),
                    )

    from subnetwork_probing.train import iterative_correspondence_from_mask
    corr, _ = iterative_correspondence_from_mask(nodes_to_mask=nodes_to_mask, model=model)

    # For all heads...
    for layer_idx, head_idx in all_nodes:
        for letter in "qkv":
            # remove input -> head connection
            edge_to = corr.edges[f"blocks.{layer_idx}.hook_{letter}_input"][TorchIndex([None, None, head_idx])]
            edge_to[f"blocks.0.hook_resid_pre"][TorchIndex([None])].present = False

            # Remove all other_head->this_head connections in the circuit
            for layer_from in range(layer_idx):
                for head_from in range(12):
                    edge_to[f"blocks.{layer_from}.attn.hook_result"][TorchIndex([None, None, head_from])].present = False

            # Remove connection from this head to the output
            corr.edges["blocks.11.hook_resid_post"][TorchIndex([None])][f"blocks.{layer_idx}.attn.hook_result"][TorchIndex([None, None, head_idx])].present = False


    special_connections: set[Conn] = {
        Conn("INPUT", "previous token", ("q", "k", "v")),
        Conn("INPUT", "duplicate token", ("q", "k", "v")),
        Conn("INPUT", "non-s inhibition", ("q",)),
        Conn("INPUT", "country-capital mover", ("k", "v")),
        Conn("INPUT", "negative name mover", ("k", "v")),
        Conn("INPUT", "name mover", ("k", "v")),
        Conn("INPUT", "backup name mover", ("k", "v")),
        Conn("previous token", "induction", ("k", "v")),
        Conn("induction", "non-s inhibition", ("k", "v")),
        Conn("duplicate token", "non-s inhibition", ("k", "v")),
        Conn("non-s inhibition", "negative name mover", ("q",)),
        Conn("non-s inhibition", "name mover", ("q",)),
        Conn("non-s inhibition", "backup name mover", ("q",)),
        Conn("negative name mover", "OUTPUT", ()),
        Conn("name mover", "OUTPUT", ()),
        Conn("backup name mover", "OUTPUT", ()),
        Conn("country-capital mover", "OUTPUT", ()),
    }

    for conn in special_connections:
        if conn.inp == "INPUT":
            idx_from = [(-1, "blocks.0.hook_resid_pre", TorchIndex([None]))]
            for mlp_layer_idx in range(12):
                idx_from.append((mlp_layer_idx, f"blocks.{mlp_layer_idx}.hook_mlp_out", TorchIndex([None])))
        else:
            idx_from = [(layer_idx, f"blocks.{layer_idx}.attn.hook_result", TorchIndex([None, None, head_idx])) for layer_idx, head_idx in CIRCUIT[conn.inp]]

        if conn.out == "OUTPUT":
            idx_to = [(13, "blocks.11.hook_resid_post", TorchIndex([None]))]
            for mlp_layer_idx in range(12):
                idx_to.append((mlp_layer_idx, f"blocks.{mlp_layer_idx}.hook_mlp_in", TorchIndex([None])))
        else:
            idx_to = [
                (layer_idx, f"blocks.{layer_idx}.hook_{letter}_input", TorchIndex([None, None, head_idx]))
                for layer_idx, head_idx in CIRCUIT[conn.out]
                for letter in conn.qkv
            ]

        for layer_from, layer_name_from, which_idx_from in idx_from:
            for layer_to, layer_name_to, which_idx_to in idx_to:
                if layer_to > layer_from:
                    corr.edges[layer_name_to][which_idx_to][layer_name_from][which_idx_from].present = True

    ret =  OrderedDict({(t[0], t[1].hashable_tuple, t[2], t[3].hashable_tuple): e.present for t, e in corr.all_edges().items() if e.present})
    return ret


GROUP_COLORS = {
    "name mover": "#d7f8ee",
    "backup name mover": "#e7f2da",
    "negative name mover": "#fee7d5",
    "country-capital mover": "#ececf5",
    "non-s inhibition": "#f9d9d7",
    "induction": "#fff6db",
    "duplicate token": "#fad6e9", 
    "previous token": "#f9ecd7",
}
MLP_COLOR = "#f0f0f0"

def kbicr_group_colorscheme():
    assert set(GROUP_COLORS.keys()) == set(CIRCUIT.keys())

    scheme = {
        "embed": "#cbd5e8",
        "<resid_post>": "#fff2ae",
    }

    for i in range(12):
        scheme[f"<m{i}>"] = MLP_COLOR

    for k, heads in CIRCUIT.items():
        for (layer, head) in heads:
            for qkv in ["", "_q", "_k", "_v"]:
                scheme[f"<a{layer}.{head}{qkv}>"] = GROUP_COLORS[k]

    for layer in range(12):
        scheme[f"<m{layer}>"] = "#f0f0f0"
    return scheme

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
