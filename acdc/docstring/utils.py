import dataclasses
from functools import partial
from types import NoneType
import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Callable, Dict
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
import warnings
import networkx as nx
import acdc.docstring.prompts as prompts
from acdc.acdc_utils import (
    MatchNLLMetric,
    make_nd_dict,
    TorchIndex,
    Edge,
    EdgeType,
    negative_log_probs,
    shuffle_tensor,
)  # these introduce several important classes !!!
from acdc import HookedTransformer
from acdc.acdc_utils import kl_divergence


@dataclasses.dataclass(frozen=False)
class AllDocstringThings:
    tl_model: HookedTransformer
    validation_metric: Callable[[torch.Tensor], torch.Tensor]
    validation_data: torch.Tensor
    validation_labels: torch.Tensor
    validation_mask: NoneType
    validation_patch_data: torch.Tensor
    test_metrics: dict[str, Any]
    test_data: torch.Tensor
    test_labels: torch.Tensor
    test_mask: NoneType
    test_patch_data: torch.Tensor


def get_all_docstring_things(
    num_examples,
    seq_len,
    device,
    metric_name="kl_div",
    dataset_version="random_random",
    correct_incorrect_wandb=True,
) -> AllDocstringThings:
    tl_model = HookedTransformer.from_pretrained(
        "attn-only-4l",
        use_global_cache=True,
    )
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    tl_model.to(device)

    docstring_ind_prompt_kwargs = dict(
        n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1, n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2
    )

    raw_prompts = [
        prompts.docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=i)
        for i in range(num_examples * 2)
    ]
    batched_prompts = prompts.BatchedPrompts(prompts=raw_prompts, model=tl_model)
    toks_int_values = batched_prompts.clean_tokens
    toks_int_values_other = batched_prompts.corrupt_tokens[dataset_version]
    toks_int_labels = batched_prompts.correct_tokens.squeeze(-1)
    toks_int_wrong_labels = batched_prompts.wrong_tokens
    assert toks_int_labels.ndim == 1
    assert toks_int_wrong_labels.ndim == 2

    validation_data = toks_int_values[:num_examples]
    validation_labels = toks_int_labels[:num_examples]
    validation_wrong_labels = toks_int_wrong_labels[:num_examples]
    validation_mask = None
    validation_patch_data = toks_int_values_other[:num_examples]

    test_data = toks_int_values[num_examples:]
    test_labels = toks_int_labels[num_examples:]
    test_wrong_labels = toks_int_wrong_labels[num_examples:]
    test_mask = None
    test_patch_data = toks_int_values_other[num_examples:]

    with torch.no_grad():
        base_validation_logprobs = F.log_softmax(tl_model(validation_data)[:, -1], dim=-1)
        base_test_logprobs = F.log_softmax(tl_model(test_data)[:, -1], dim=-1)
        assert len(base_validation_logprobs.shape) == 2, base_validation_logprobs.shape

    def raw_docstring_metric(
        logits: torch.Tensor,
        correct_labels: torch.Tensor,
        wrong_labels: torch.Tensor,
        log_correct_incorrect_wandb: bool = False,
    ):
        """With neg sign so we minimize this"""

        correct_logits = logits[torch.arange(len(logits)), -1, correct_labels]
        incorrect_logits = logits[torch.arange(len(logits)).unsqueeze(-1), -1, wrong_labels]

        if log_correct_incorrect_wandb:
            wandb.log(
                {
                    "correct_logits": correct_logits.mean().item(),
                    "incorrect_logits": incorrect_logits.max(dim=-1).values.mean().item(),
                }
            )

        # note neg sign!!!
        return -(correct_logits.mean() - incorrect_logits.max(dim=-1).values.mean())

    def ldgz_docstring_metric(
        logits: torch.Tensor,
        correct_labels: torch.Tensor,
        wrong_labels: torch.Tensor,
    ):
        """Logit diff greater zero fraction (with neg sign)"""
        pos_logits = logits[:, -1, :]
        max_correct, _ = torch.gather(pos_logits, index=correct_labels[..., None], dim=1).max(dim=1)
        max_wrong, _ = torch.gather(pos_logits, index=wrong_labels, dim=1).max(dim=1)
        return -((max_correct - max_wrong > 0).sum() / len(max_correct))

    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
        )
    elif metric_name == "docstring_metric":
        validation_metric = partial(
            raw_docstring_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
            log_correct_incorrect_wandb=correct_incorrect_wandb,
        )
    elif metric_name == "docstring_stefan":
        validation_metric = partial(
            ldgz_docstring_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
        )
    elif metric_name == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            last_seq_element_only=True,
        )
    elif metric_name == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
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
        "docstring_metric": partial(
            raw_docstring_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
            log_correct_incorrect_wandb=correct_incorrect_wandb,
        ),
        "docstring_stefan": partial(
            ldgz_docstring_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            last_seq_element_only=True,
        ),
        "match_nll": MatchNLLMetric(
            labels=test_labels,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
        ),
    }

    return AllDocstringThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data,
    )
