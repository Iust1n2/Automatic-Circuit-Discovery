import json
import torch
from graphviz import Source
import re
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    TypeVar,
    Iterable,
    Set,
)
from acdc.hybridretrieval.utils import CIRCUIT, AllDataThings
from acdc.acdc_utils import get_node_stats, get_present_nodes
from subnetwork_probing.train import iterative_correspondence_from_mask
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCInterpNode import parse_interpnode, heads_to_nodes_to_mask
from acdc.TLACDCEdge import (
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment

import os
from pathlib import Path

__file__ = os.getcwd()

ACDC_DIR = Path(__file__).resolve() / "acdc" / "hybridretrieval" / "acdc_results" / "kbicr"
print(ACDC_DIR)

SP_DIR = Path(__file__).resolve() / "subnetwork_probing" / "results"
print(SP_DIR)

@dataclass(frozen=True)
class AcdcRunCandidate:
    threshold: float
    steps: int
    run: str
    score_d: dict
    corr: TLACDCCorrespondence


def parse_gv_files(acdc_dir):
    """Parses the latest Graphviz .gv file from each ACDC run directory, extracting nodes and storing them by run name and threshold."""

    full_project_path = os.path.abspath(acdc_dir)
    try:
        # Get list of run directories
        run_dirs = [d for d in os.listdir(full_project_path) if os.path.isdir(os.path.join(full_project_path, d))]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}

    # Filter run directories using a pattern (e.g., logit_diff_*)
    pattern = re.compile(r'logit_diff_(\d+\.\d+)')
    run_dirs = [d for d in run_dirs if pattern.search(d)]
    
    print(f"loading {len(run_dirs)} runs from local path {full_project_path}")

    # Dictionary to store nodes for each run
    run_node_dict = {}

    for run in run_dirs:
        run_path = os.path.join(full_project_path, run)

        # Get the threshold for this run (you have a separate function for this)
        threshold = extract_acdc_threshold_from_path(run)  # Assuming you have this function implemented

        # Get list of .gv files matching the prefix "img_new_" followed by a number and ending in ".gv"
        gv_files = [f for f in os.listdir(run_path) if re.match(r'img_new_\d+\.gv', f)]
        
        if not gv_files:
            print(f"No .gv files found in {run}")
            continue

        # Sort .gv files by the number in their filename and select the last one (final file)
        gv_files_sorted = sorted(gv_files, key=lambda x: int(re.search(r'img_new_(\d+)\.gv', x).group(1)))
        latest_gv_file = gv_files_sorted[-1]

        gv_file_path = os.path.join(run_path, latest_gv_file)
        print(f"Parsing final .gv file for run {run} with threshold {threshold}: {gv_file_path}")

        try:
            with open(gv_file_path, 'r') as file:
                gv_content = file.read()

            # Extract only node names like "<m10>" (without considering edges)
            node_names = re.findall(r'\"<([^>]+)>\"(?=\s*\[)', gv_content)

            # Store node names in the dictionary with run name and threshold as key
            run_key = f"ACDC_logit_diff_{threshold}"
            run_node_dict[run_key] = node_names

        except (FileNotFoundError, IOError) as e:
            print(f"Error opening .gv file for run {run}: {e}")
            continue

    return run_node_dict


def extract_acdc_threshold_from_path(acdc_run):
    """Extract threshold from the directory name"""
    match = re.search(r'logit_diff_(\d+\.\d+)', acdc_run)
    if match:
        return float(match.group(1))
    return None

def extract_sp_threshold_from_path(sp_run):
    """Extract threshold from the directory name"""
    match = re.search(r'hybrid-retrieval_lr_0\.01_lambda_reg_(\d+(\.\d+)?)', sp_run)
    if match:
        return float(match.group(1))
    return None

def all_test_fns(things: AllDataThings, data: torch.Tensor) -> dict[str, float]:
    """Placeholder function to call test metrics on the data in the parse_acdc_results function."""
    return {f"test_{name}": fn(data).item() for name, fn in things.test_metrics.items()}

def parse_acdc_results(exp: TLACDCExperiment, things: AllDataThings, acdc_dir: str):
    """Parses the ACDC results directory to extract adjacencies as TLACDCCorrespondence objects and runs test metrics."""
    all_candidates = []
    # Load runs from the local directory
    full_project_path = os.path.abspath(acdc_dir)
    try:
        run_dirs = [d for d in os.listdir(full_project_path) if os.path.isdir(os.path.join(full_project_path, d))]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    pattern = re.compile(r'logit_diff_(\d+\.\d+)')
    run_dirs = [d for d in run_dirs if pattern.search(d)]
    
    print(f"loading {len(run_dirs)} runs from local path {full_project_path}")
    threshold_to_run_map: dict[float, AcdcRunCandidate] = {}

    def add_run_for_processing(candidate: AcdcRunCandidate):
        if candidate.threshold not in threshold_to_run_map:
            threshold_to_run_map[candidate.threshold] = candidate
        else:
            if candidate.steps > threshold_to_run_map[candidate.threshold].steps:
                threshold_to_run_map[candidate.threshold] = candidate

    for run in run_dirs:
        threshold = extract_acdc_threshold_from_path(run)

        score_d = {
            "score": threshold
        }  

        corr = deepcopy(exp.corr)
        try:
            edges_artifact_path = os.path.join(full_project_path, run, 'edges.pth')
        except KeyError:
            print(f"edges.pth not found for run {run}")
            continue

        if os.path.exists(edges_artifact_path):
            with open(edges_artifact_path, "rb") as f:
                edges_pth = torch.load(edges_artifact_path)
    
        for (n_to, idx_to, n_from, idx_from) in edges_pth:
                n_to = n_to.replace("hook_resid_mid", "hook_mlp_in")
                n_from = n_from.replace("hook_resid_mid", "hook_mlp_in")
                idx_from, idx_to = TorchIndex(idx_from), TorchIndex(idx_to)
                
                # all_edges[n_to][idx_to][n_from][idx_from].present = True
                corr.edges[n_to][idx_to][n_from][idx_from].present = True

        add_run_for_processing(AcdcRunCandidate(
            threshold=threshold,
            steps=score_d["steps"],
            run=run,
            score_d=score_d,
            corr=corr,
        ))
    
    all_candidates.extend(list(threshold_to_run_map.values()))

    for candidate in all_candidates:
        exp.setup_corrupted_cache()
        test_metrics = exp.call_metric_with_corr(candidate.corr, all_test_fns, things.test_data)
        candidate.score_d.update(test_metrics)
        print(f"Added run with threshold={candidate.threshold}, n_edges={candidate.corr.count_no_edges()}")

    corrs = [(candidate.corr, candidate.score_d) for candidate in all_candidates]
    return corrs

def parse_sp_results(
    model,
    things: AllDataThings,
    project_path: str = None,
    run_filter: Optional[Callable[[Any], bool]] = None,
    clip: Optional[int] = None,
):
    # Load runs from the local directory
    full_project_path = os.path.abspath(project_path)
    try:
        run_dirs = [d for d in os.listdir(full_project_path) if os.path.isdir(os.path.join(full_project_path, d))]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    pattern = re.compile(r'hybrid-retrieval_lr_0\.01_lambda_reg_(\d+(\.\d+)?)')
    run_dirs = [d for d in run_dirs if pattern.search(d)]
    
    if run_filter:
        run_dirs = list(filter(run_filter, run_dirs))
    run_dirs = run_dirs[:clip]
    
    print(f"loading {len(run_dirs)} runs from local path {full_project_path}")
    
    if run_filter:
        run_dirs = list(filter(run_filter, run_dirs))
    run_dirs = run_dirs[:clip]
    
    corrs = []
    corr, head_parents = None, None

    for run in run_dirs:

        threshold = extract_sp_threshold_from_path(run)
        score_d = {}
        score_d['threshold'] = threshold

        results_path = os.path.join(full_project_path, run, 'results.json')
        mask_scores_path = os.path.join(full_project_path, run, 'mask_scores.json')
        if not os.path.exists(mask_scores_path):
            print(f"Metrics file not found for run {run}")
            continue
        if not os.path.exists(results_path):
            print(f"Results file not found for run {run}")
            continue

        with open(results_path, 'r') as f:
            d = json.load(f)
            data = d["nodes_to_mask"]
        
        with open(mask_scores_path, 'r') as f:
            d = json.load(f)
            masked_nodes = d['x']
            mask_scores = d['y']
            

        if things is None:
            return [
                (None, {"score": threshold})
            ]

        try:
            nodes_to_mask_strings = data
        except (KeyError, FileNotFoundError):
            print(f"Error: Could not load nodes_to_mask from {run}")
        # try: # this errors, not enough values to unpack (expected 2, got 1)
        #     nodes_to_mask_strings = masked_nodes 
        # except(KeyError, FileNotFoundError):
        #     print(f"Error: Could not load nodes_to_mask from {run}")
            continue

        # Parse the nodes_to_mask strings
        nodes_to_mask = [parse_interpnode(s) for s in nodes_to_mask_strings]

        # Compute the correspondence from the masked nodes
        corr, head_parents = iterative_correspondence_from_mask(
            model=model,
            nodes_to_mask=nodes_to_mask,
            use_pos_embed=None,
            corr=None,
            head_parents=None
        )

        # Extract score and additional information
        score_d = {}
        score_d["score"] = threshold
        score_d["nodes"] = nodes_to_mask_strings

        # Append the correspondence and scores to the list
        corrs.append((deepcopy(corr), score_d))

    return corrs

def parse_sp_manually(sp_dir: str): 
    full_project_path = os.path.abspath(sp_dir)
    try:
        run_dirs = [d for d in os.listdir(full_project_path) if os.path.isdir(os.path.join(full_project_path, d))]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    pattern = re.compile(r'hybrid-retrieval_lr_0\.01_lambda_reg_(\d+(\.\d+)?)')
    run_dirs = [d for d in run_dirs if pattern.search(d)]
    
    print(f"loading {len(run_dirs)} runs from local path {full_project_path}")

    run_node_dict = {}

    for run in run_dirs:
        run_path = os.path.join(full_project_path, run)
        
        threshold = extract_sp_threshold_from_path(run)

        results_path = os.path.join(full_project_path, run, 'results.json')
        mask_scores_path = os.path.join(full_project_path, run, 'mask_scores.json')
        if not os.path.exists(mask_scores_path):
            print(f"Metrics file not found for run {run}")
            continue
        if not os.path.exists(results_path):
            print(f"Results file not found for run {run}")
            continue

        with open(results_path, 'r') as f:
            d = json.load(f)
            data = d["nodes_to_mask"]
            mask_val_dict = d["mask_val_dict"]
            n_nodes = d["number_of_nodes"]

        with open(mask_scores_path, 'r') as f:
            d = json.load(f)
            masked_nodes = d['x']
            mask_scores = d['y']

        try:
            nodes_to_mask_strings = data
        except (KeyError, FileNotFoundError):
            print(f"Error: Could not load nodes_to_mask from {run}")
            continue
        print(f"Parsing results.json for run {run} with threshold {threshold}: {results_path}")

        # print(f"Number of nodes: {n_nodes}")
        # print(f"Mask values: {len(mask_val_dict)}")
        # print(f"Nodes to mask: {len(nodes_to_mask_strings)}")

        # Filter mask_val_dict to keep only non-zero entries
        non_zero_entries = [node for node, value in mask_val_dict.items() if value != 0]
        
        # second assertion is because n_nodes include resid_mid activations and mask_val_dict doen't
        # assert n_nodes == len(data), f"Expected {n_nodes} nodes, but got {len(data)}"
        # assert len(non_zero_entries) == n_nodes, f"Number of non-zero entries ({len(non_zero_entries)}) does not match n_nodes ({n_nodes})"
        
        # Store node names in the dictionary with run name as key
        run_key = f"SP_lambda_reg_{threshold}"
        run_node_dict[run_key] = non_zero_entries

        # Debug: Check the filtered non-zero entries
        print(f"Filtered non-zero nodes for run {run_key}:")
        
        # # Parse the nodes_to_mask strings
        # nodes_to_mask = [parse_interpnode(s) for s in nodes_to_mask_strings]

        # # Compute the correspondence from the masked nodes
        # _, head_parents = iterative_correspondence_from_mask(
        #     model=model,
        #     nodes_to_mask=nodes_to_mask,
        #     use_pos_embed=None,
        #     corr=None,
        #     head_parents=None
        # )

    #    # Modify head_parents to format (Layer, Head, q/k/v)
    #     formatted_head_parents = {}
    #     for head_key, parent_layer in nodes_to_mask:
    #         head_name, head_index = head_key  # e.g., ('blocks.0.attn.hook_result', [:, :, 0])

    #         # Extract layer, head, and q/k/v from the head_key string
    #         match = re.match(r'blocks\.(\d+)\.attn\.hook_(q|k|v)', head_name)
    #         if match:
    #             layer = int(match.group(1))
    #             qkv = match.group(2)  # Extract q, k, or v

    #             head_number = head_index[-1]  # Extract the head number from [:, :, Head]
    #             formatted_head_parents[(layer, head_number, qkv)] = parent_layer

        # Store node names in the dictionary with run name and threshold as key
        # run_key = f"{threshold}"
        # run_node_dict[run_key] = formatted_head_parents
    
    return run_node_dict


def simplify_recovered_nodes(recovered_nodes, is_simplified=True):
    """
    Simplifies recovered nodes by extracting either (layer, head) pairs or keeps the full format (layer.head.q/k/v).
    Filters out MLP nodes (those starting with 'm').

    Args:
        recovered_nodes (set): The set of nodes recovered from the .gv files or SP runs.
        is_simplified (bool): If True, extract only (layer, head) pairs. If False, keep the full format (layer.head.q/k/v).

    Returns:
        set: Simplified or non-simplified set of nodes.
    """
    processed_nodes = set()

    for node in recovered_nodes:
        # Skip MLP nodes (e.g., 'm1', 'm2')
        if node.startswith('m'):
            continue
        
        # Handle ACDC gv format (e.g., 'a9.9_q', 'a9.9_v')
        if node.startswith('a'):
            match = re.match(r'a(\d+)\.(\d+)_([qkv])', node)
            if match:
                layer = int(match.group(1))
                head = int(match.group(2))
                qkv = match.group(3)
                if is_simplified:
                    processed_nodes.add((layer, head))  # Simplified to (layer, head)
                else:
                    processed_nodes.add(f"{layer}.{head}.{qkv}")  # Keep format like '9.9.q'
        
        # Handle SP format (e.g., '9.9.q', '9.9.k', '9.9.v')
        elif re.match(r'(\d+)\.(\d+)\.[qkv]', node):
            match = re.match(r'(\d+)\.(\d+)\.([qkv])', node)
            if match:
                layer = int(match.group(1))
                head = int(match.group(2))
                qkv = match.group(3)

                if is_simplified:
                    processed_nodes.add((layer, head))  # Simplified to (layer, head)
                else:
                    processed_nodes.add(f"{layer}.{head}.{qkv}")  # Keep format like '9.9.q'

    return processed_nodes


def get_run_stats(canonical_circuit: dict, acdc_dir: str, sp_dir: str, is_simplified=True):
    """
    Compares recovered circuits from ACDC and SP with a canonical circuit and computes various overlap statistics.

    Args:
        canonical_circuit (dict): 
            The canonical circuit as a dictionary of circuit categories (e.g., "name mover") with the associated heads (Layer, Head).
        acdc_dir (str): 
            The directory where ACDC Graphviz (.gv) files are stored.
        sp_dir (str): 
            The directory where SP results are stored.
        is_simplified (bool): 
            If True, simplifies the recovered nodes by extracting only (Layer, Head) pairs for attention heads.
            If False, retains the full format (layer.head.q/k/v) for comparing circuits with themselves.

    Returns (if is_simplified=True):
        acdc_comparison_stats (dict): 
            A dictionary containing ACDC run statistics, where each entry contains:
            - number of recovered nodes: Number of nodes recovered by ACDC.
            - intersection size: Size of the intersection between ACDC and canonical circuit nodes.
            - jaccard index: Jaccard index of ACDC with the canonical circuit.

        sp_comparison_stats (dict): 
            A dictionary containing SP run statistics, where each entry contains:
            - number of recovered nodes: Number of nodes recovered by SP.
            - intersection size: Size of the intersection between SP and canonical circuit nodes.
            - jaccard index: Jaccard index of SP with the canonical circuit.

    Returns (if is_simplified=False):
        acdc_comparison_stats (dict):
            A dictionary containing ACDC run statistics, where each entry contains:
            - number of recovered nodes: Number of nodes recovered by ACDC.
            - recovered nodes: List of nodes recovered by ACDC.

        sp_comparison_stats (dict):
            A dictionary containing SP run statistics, where each entry contains:
            - number of recovered nodes: Number of nodes recovered by SP.
            - recovered nodes: List of nodes recovered by SP.
        
        acdc_vs_sp_comparison_stats (dict): 
            A dictionary comparing ACDC and SP runs with both each other and the canonical circuit, where each entry contains:
            - no. of acdc recovered nodes: Number of nodes recovered by ACDC.
            - no. of sp recovered nodes: Number of nodes recovered by SP.
            - list of acdc nodes: List of nodes recovered by ACDC.
            - list of sp nodes: List of nodes recovered by SP.
            - overlap intersection size: Size of the intersection between ACDC and SP nodes.
            - overlap intersection list: List of nodes present in both ACDC and SP.
            - overlap jaccard index: Jaccard index between ACDC and SP nodes.
            - ACDC \ SP: List of nodes present in ACDC but not in SP.
            - SP \ ACDC: List of nodes present in SP but not in ACDC.
            - acdc canonical intersection size: Size of the intersection between ACDC and canonical circuit nodes.
            - acdc canonical intersection: List of nodes in both ACDC and canonical circuit.
            - sp canonical intersection size: Size of the intersection between SP and canonical circuit nodes.
            - sp canonical intersection: List of nodes in both SP and canonical circuit.
            - acdc \ sp canonical intersection size: Size of the intersection between ACDC \ SP nodes and canonical circuit nodes.
            - acdc \ sp canonical intersection: List of nodes in both ACDC \ SP and canonical circuit.
            - sp \ acdc canonical intersection size: Size of the intersection between SP \ ACDC nodes and canonical circuit nodes.
            - sp \ acdc canonical intersection: List of nodes in both SP \ ACDC and canonical circuit.
            - acdc canonical jaccard: Jaccard index of ACDC with the canonical circuit.
            - sp canonical jaccard: Jaccard index of SP with the canonical circuit.
            - acdc \ sp canonical jaccard: Jaccard index of ACDC \ SP with the canonical circuit.
            - sp \ acdc canonical jaccard: Jaccard index of SP \ ACDC with canonical circuit.
    """

    # Extract all nodes from the canonical circuit
    all_groups_of_nodes = [group for _, group in canonical_circuit.items()]
    true_nodes = set(node for group in all_groups_of_nodes for node in group)

    # Parse recovered nodes for each ACDC run
    acdc_adjacencies = parse_gv_files(acdc_dir=acdc_dir)
    sp_adjacencies = parse_sp_manually(sp_dir=sp_dir)

    acdc_comparison_stats = {}
    sp_comparison_stats = {}
    acdc_vs_sp_comparison_stats = {}

    if is_simplified: 
        for run_name, acdc_recovered_nodes in acdc_adjacencies.items():
            
            # Simplify recovered nodes to (layer, head) format
            acdc_recovered_nodes = simplify_recovered_nodes(acdc_recovered_nodes, is_simplified=is_simplified) 

            print(f"Run: {run_name}")
            print(f"Canonical nodes: {true_nodes}")
            print(f"Recovered (simplified) nodes: {acdc_recovered_nodes}")

            # Compute intersection and Jaccard index
            intersection = compute_manual_intersection(true_nodes, acdc_recovered_nodes)
            jaccard_index = compute_manual_jaccard_index(true_nodes, acdc_recovered_nodes)

            acdc_comparison_stats[run_name] = {
                ''
                'number of canonical nodes': len(true_nodes),
                'number of recovered nodes': len(acdc_recovered_nodes),
                'intersection size': len(intersection),
                'jaccard index': jaccard_index
            }

        for run_name, sp_recovered_nodes in sp_adjacencies.items():
            
            # Simplify recovered nodes to (layer, head) format
            sp_recovered_nodes = simplify_recovered_nodes(sp_recovered_nodes, is_simplified=is_simplified) 

            print(f"Run: {run_name}")
            print(f"Canonical nodes: {true_nodes}")
            print(f"Recovered (simplified) nodes: {sp_recovered_nodes}")

            # Compute intersection and Jaccard index
            intersection = compute_manual_intersection(true_nodes, sp_recovered_nodes)
            jaccard_index = compute_manual_jaccard_index(true_nodes, sp_recovered_nodes)

            sp_comparison_stats[run_name] = {
                ''
                'number of canonical nodes': len(true_nodes),
                'number of recovered nodes': len(sp_recovered_nodes),
                'intersection size': len(intersection),
                'jaccard index': jaccard_index
            }

    else:
        # Prepare and sort ACDC and SP scores by the number of recovered nodes
        for run_name, acdc_recovered_nodes in acdc_adjacencies.items():
            recovered_nodes = simplify_recovered_nodes(acdc_recovered_nodes, is_simplified=False)
            acdc_comparison_stats[run_name] = {
                'number of recovered nodes': len(recovered_nodes),
                'recovered nodes': recovered_nodes,
            }

        for run_name, sp_recovered_nodes in sp_adjacencies.items():
            recovered_nodes = simplify_recovered_nodes(sp_recovered_nodes, is_simplified=False)
            sp_comparison_stats[run_name] = {
                'number of recovered nodes': len(recovered_nodes),
                'recovered nodes': recovered_nodes,
            }

        # Sort both ACDC and SP runs by the number of recovered nodes
        sorted_acdc_scores = dict(sorted(acdc_comparison_stats.items(), key=lambda x: x[1]['number of recovered nodes']))
        sorted_sp_scores = dict(sorted(sp_comparison_stats.items(), key=lambda x: x[1]['number of recovered nodes']))

        # # Compare sorted ACDC and SP runs one by one in the sorted order
        # for (acdc_run_name, acdc_data), (sp_run_name, sp_data) in zip(sorted_acdc_scores.items(), sorted_sp_scores.items()):
        #     acdc_recovered_nodes = acdc_data['recovered nodes']
        #     sp_recovered_nodes = sp_data['recovered nodes']

        #     # Compute intersection and Jaccard index between ACDC and SP nodes
        #     results = compare_manual_circuits(acdc_recovered_nodes, sp_recovered_nodes)
        #     intersection = results['intersection size']
        #     jaccard_index = results['jaccard index']

        #     acdc_vs_sp_comparison_stats[f"{acdc_run_name} vs {sp_run_name}"] = {
        #         'no. of acdc recovered nodes': len(acdc_recovered_nodes),
        #         'no. of sp recovered nodes': len(sp_recovered_nodes),
        #         'intersection size': intersection,
        #         'jaccard index': jaccard_index
        #     }
        
        # Compare sorted ACDC and SP runs one ACDC with all SP in the sorted order
        for (acdc_run_name, acdc_data) in sorted_acdc_scores.items():
            acdc_recovered_nodes = acdc_data['recovered nodes']
            for (sp_run_name, sp_data) in sorted_sp_scores.items():
                sp_recovered_nodes = sp_data['recovered nodes']

                # Compute intersection and Jaccard index between ACDC and SP nodes
                results = compare_manual_circuits(acdc_recovered_nodes, sp_recovered_nodes)
                overlap_intersection = results['intersection size']
                overlap_intersection_list = results['intersection list']
                overlap_jaccard_index = results['jaccard index']

                # Compute ACDC \ SP and SP \ ACDC
                differences = calculate_set_differences(acdc_recovered_nodes, sp_recovered_nodes, is_simplified=True)
                acdc_minus_sp = differences['acdc_minus_sp']
                sp_minus_acdc = differences['sp_minus_acdc']

                # Simplify recovered nodes to (Layer, Head) format for comparing with the canonical circuit
                acdc_simplified_recovered_nodes = simplify_recovered_nodes(acdc_recovered_nodes, is_simplified=True) 
                sp_simplified_recovered_nodes = simplify_recovered_nodes(sp_recovered_nodes, is_simplified=True) 
                
                # Intersection with canonical circuit
                acdc_intersection = compute_manual_intersection(true_nodes, acdc_simplified_recovered_nodes)
                sp_intersection = compute_manual_intersection(true_nodes, sp_simplified_recovered_nodes)
                acdc_minus_sp_intersection = compute_manual_intersection(true_nodes, acdc_minus_sp)
                sp_minus_acdc_intersection = compute_manual_intersection(true_nodes, sp_minus_acdc)

                # Refactor intersection sets to tuples for better readability
                acdc_intersection = [tuple(node) for node in acdc_intersection]
                sp_intersection = [tuple(node) for node in sp_intersection]
                acdc_minus_sp_intersection = [tuple(node) for node in acdc_minus_sp_intersection]
                sp_minus_acdc_intersection = [tuple(node) for node in sp_minus_acdc_intersection]

                # Jaccard index with canonical circuit
                acdc_jaccard_index = compute_manual_jaccard_index(true_nodes, acdc_simplified_recovered_nodes)
                sp_jaccard_index = compute_manual_jaccard_index(true_nodes, sp_simplified_recovered_nodes)
                acdc_minus_sp_jaccard_index = compute_manual_jaccard_index(true_nodes, acdc_minus_sp)
                sp_minus_acdc_jaccard_index = compute_manual_jaccard_index(true_nodes, sp_minus_acdc)

                acdc_vs_sp_comparison_stats[f"{acdc_run_name} vs {sp_run_name}"] = {
                    'canonical nodes': sorted(true_nodes),
                    'no. of acdc recovered nodes': len(acdc_recovered_nodes),
                    'no. of sp recovered nodes': len(sp_recovered_nodes),
                    'list of acdc nodes': list(acdc_recovered_nodes),
                    'list of sp nodes': list(sp_recovered_nodes),
                    'overlap intersection size': overlap_intersection,
                    'overlap intersection list': overlap_intersection_list,
                    'overlap jaccard index': overlap_jaccard_index,
                    'ACDC \ SP': sorted(acdc_minus_sp),
                    'SP \ ACDC': sorted(sp_minus_acdc),
                    'acdc canonical intersection size': len(acdc_intersection),
                    'acdc canonical intersection': sorted(acdc_intersection),
                    'sp canonical intersection size': len(sp_intersection),
                    'sp canonical intersection': sorted(sp_intersection),
                    'acdc \ sp canonical intersection size': len(acdc_minus_sp_intersection),
                    'acdc \ sp canonical intersection': sorted(acdc_minus_sp_intersection),
                    'sp \ acdc canonical intersection size': len(sp_minus_acdc_intersection),
                    'sp \ acdc canonical intersection': sorted(sp_minus_acdc_intersection),
                    'acdc canonical jaccard': acdc_jaccard_index,
                    'sp canonical jaccard': sp_jaccard_index,
                    'acdc \ sp canonical jaccard': acdc_minus_sp_jaccard_index,
                    'sp \ acdc canonical jaccard': sp_minus_acdc_jaccard_index,
                }

    return acdc_comparison_stats, sp_comparison_stats, acdc_vs_sp_comparison_stats

# ---------- Helper functions for circuit statistics ----------
def compute_intersection(ground_truth, recovered_circuit):
    """Computes the intersection of circuits in TLACDCCorespondence format."""
    ground_truth_nodes, _ = get_present_nodes(ground_truth)

    recovered_nodes, _ = get_present_nodes(recovered_circuit)
    return recovered_nodes.intersection(ground_truth_nodes)

def compute_jaccard_index(ground_truth, recovered_circuit):
    """Computes the Jaccard Index between circuits in TLACDCCorespondence format."""
    ground_truth_nodes, _ = get_present_nodes(ground_truth)

    recovered_nodes, _ = get_present_nodes(recovered_circuit)
    intersection = recovered_nodes.intersection(ground_truth_nodes)
    union = recovered_nodes.union(ground_truth_nodes)
    if not union:
        return 0
    return len(intersection) / len(union)

def compare_circuits(canonical_circuit, algorithm_circuit: TLACDCCorrespondence):
    """Compares circuits in TLACDCCorespondence format using both Intersection and Jaccard Index."""
    intersection = compute_intersection(canonical_circuit, algorithm_circuit)
    jaccard = compute_jaccard_index(canonical_circuit, algorithm_circuit)
    return {
        'intersection size': len(intersection),
        'jaccard index': jaccard
    }

def compute_manual_intersection(canonical_nodes, parsed_nodes):
    """Computes the intersection of manually parsed circuits."""
    intersection = canonical_nodes.intersection(parsed_nodes)
    return intersection

def compute_manual_jaccard_index(canonical_nodes, parsed_nodes):
    """Computes the intersection of manually parsed circuits."""
    intersection = canonical_nodes.intersection(parsed_nodes)
    union = canonical_nodes.union(parsed_nodes)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def calculate_set_differences(acdc_recovered_nodes, sp_recovered_nodes, is_simplified=True):
    """
    Calculate the set differences ACDC \ SP and SP \ ACDC in simplified form (Layer, Head).
    
    Args:
        acdc_recovered_nodes (set): The set of recovered nodes from ACDC.
        sp_recovered_nodes (set): The set of recovered nodes from SP.
        is_simplified (bool): Whether to simplify the nodes to (Layer, Head) format instead of (layer.head.q/k/v).

    Returns:
        dict: Contains the set differences sorted in ascending order.
    """
    # Simplify recovered nodes for both ACDC and SP
    acdc_simplified_recovered_nodes = simplify_recovered_nodes(acdc_recovered_nodes, is_simplified=is_simplified)
    sp_simplified_recovered_nodes = simplify_recovered_nodes(sp_recovered_nodes, is_simplified=is_simplified)

    # Compute ACDC \ SP and SP \ ACDC
    acdc_minus_sp = acdc_simplified_recovered_nodes - sp_simplified_recovered_nodes
    sp_minus_acdc = sp_simplified_recovered_nodes - acdc_simplified_recovered_nodes

    return {
        'acdc_minus_sp': sorted(acdc_minus_sp),
        'sp_minus_acdc': sorted(sp_minus_acdc)
    }

def compare_manual_circuits(circuit_a, circuit_b):
    """Compares manually parsed circuits using both Intersection and Jaccard Index."""
    intersection = compute_manual_intersection(circuit_a, circuit_b)
    jaccard = compute_manual_jaccard_index(circuit_a, circuit_b)
    return {
        'intersection size': len(intersection),
        'intersection list': list(intersection),
        'jaccard index': jaccard
    }

def format_results(acdc_results, sp_results):
    """Formats the results from ACDC and SP runs for comparison. Produces a table-like structure with separate sections for ACDC and SP results."""
    formatted_results = {}

    # First, add ACDC results
    for run_name, acdc_stats in acdc_results.items():
        formatted_results[run_name] = {
            'no. of canonical nodes': 22,  
            'no. of acdc recovered nodes': acdc_stats.get('number of recovered nodes', 'N/A'),
            'acdc intersection size': acdc_stats.get('intersection size', 'N/A'),
            'acdc jaccard index': acdc_stats.get('jaccard index', 'N/A'),
            # 'sp_n_nodes': 'N/A',  # SP metrics will be 'N/A' here
            # 'sp_intersection_size': 'N/A',
            # 'sp_jaccard_index': 'N/A'
        }

    # Then, add SP results
    for run_name, sp_stats in sp_results.items():
        if run_name in formatted_results:
            # If the run exists in ACDC, add SP metrics to the same entry
            formatted_results[run_name].update({
                'no. of sp recovered nodes': sp_stats.get('number of recovered nodes', 'N/A'),
                'sp intersection size': sp_stats.get('intersection size', 'N/A'),
                'sp jaccard index': sp_stats.get('jaccard index', 'N/A')
            })
        else:
            # If the run is not in ACDC, create a new entry with only SP metrics
            formatted_results[run_name] = {
                # 'no. of acdc recovered nodes': 'N/A',  # ACDC metrics will be 'N/A' here
                # 'acdc intersection size': 'N/A',
                # 'acdc jaccard index': 'N/A',
                'no. of canonical nodes': 22,  
                'no. of sp recovered nodes': sp_stats.get('number of recovered nodes', 'N/A'),
                'sp intersection size': sp_stats.get('intersection size', 'N/A'),
                'sp jaccard index': sp_stats.get('jaccard index', 'N/A')
            }

    return formatted_results

# TODO add ACDC intersected with SP, ACDC \ SP, SP \ ACDC, 
# for node in list(ACDC intersected with SP):
     # if simplify_node(node) in canonical_circuit:
        # return simplify_node(node) + "*")
    # else:
    # return simplify_node(node)

# This will be as 3 columns in the table, for each run,

# Final table will be: 
# Run name | list of ACDC nodes | list of SP nodes | ACDC intersected with SP | Jacard |  ACDC \ SP | SP \ ACDC | 
