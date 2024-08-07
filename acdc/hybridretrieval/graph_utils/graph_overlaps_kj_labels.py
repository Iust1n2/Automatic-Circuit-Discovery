import networkx as nx
import os 

os.chdir('/home/iustin/Mech-Interp/Automatic-Circuit-Discovery/acdc/')
# def read_adjacency_list(file_content):
#     edges = []
#     lines = file_content.strip().split('\n')
#     for line in lines:
#         if line.strip() == "#" or not line.strip():
#             continue
#         parts = line.strip().split()
#         if len(parts) == 2:
#             node1, node2 = parts
#             edges.append((node1, node2))
#     return edges

# def read_adjacency_list(file_path):
#     edges = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             node1, node2 = line.strip().split()
#             edges.append((node1, node2))
#     return edges

def read_adjacency_list(file_path):
    edges = []
    reading_edges = False
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "#":
                reading_edges = True
                continue
            if reading_edges:
                node1, node2 = line.strip().split()
                edges.append((node1, node2))
    return edges

def add_edges_to_graph(graph, edges, graph_label):
    for node1, node2 in edges:
        if graph.has_edge(node1, node2):
            graph[node1][node2]['graphs'].add(graph_label)
        else:
            graph.add_edge(node1, node2, graphs={graph_label})

def get_edge_color(graphs):
    color_map = {
        frozenset({'J'}): 'J',
        frozenset({'K'}): 'K',
        frozenset({'M'}): 'M',
        frozenset({'J', 'K'}): 'JK',
        frozenset({'J', 'M'}): 'JM',
        frozenset({'K', 'M'}): 'KM',
        frozenset({'J', 'K', 'M'}): 'JKM'
    }
    return color_map.get(frozenset(graphs), 'grey')

def write_tgf(graph, tgf_file):
    with open(tgf_file, 'w') as f:
        node_ids = {node: idx + 1 for idx, node in enumerate(graph.nodes())}
        # Write nodes
        for node, node_id in node_ids.items():
            f.write(f"{node} {node}\n")

        # Write delimiter
        f.write("#\n")

        # Write edges
        for node1, node2, data in graph.edges(data=True):
            source = node1
            target = node2
            edge_color = get_edge_color(data['graphs'])
            f.write(f"{source} {target} {edge_color}\n")

# Read graphs from files
edges1 = read_adjacency_list('hybridretrieval/acdc_results/test/ims_hybridretrieval_indirect_0.15/ims_hybridretrieval_indirect.tgf')
edges2 = read_adjacency_list('hybridretrieval/acdc_results/test/ims_join_indirect_0.15/ims_join_indirect.tgf')
edges3 = read_adjacency_list('hybridretrieval/acdc_results/test/ims_knowledgeretrieval_indirect_0.15/ims_knowledgeretrieval_indirect.tgf')

# Create a unified graph
G = nx.DiGraph()

# Add edges from each graph to the unified graph
add_edges_to_graph(G, edges1, 'M')
add_edges_to_graph(G, edges2, 'J')
add_edges_to_graph(G, edges3, 'K')

# Write the unified graph to a TGF file
write_tgf(G, 'hybridretrieval/combined_tgf/combined_graph_indirect_kj_labels_new.tgf')
