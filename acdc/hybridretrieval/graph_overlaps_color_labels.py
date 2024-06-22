import networkx as nx
import os 

os.chdir('/home/iustin/Mech-Interp/Automatic-Circuit-Discovery/acdc/')
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
        frozenset({'J'}): 'red',
        frozenset({'K'}): 'yellow',
        frozenset({'KJ'}): 'blue',
        frozenset({'J', 'K'}): 'orange',
        frozenset({'J', 'KJ'}): 'purple',
        frozenset({'K', 'KJ'}): 'green',
        frozenset({'J', 'K', 'KJ'}): 'black'
    }
    return color_map.get(frozenset(graphs), 'grey')

def write_tgf(graph, tgf_file):
    with open(tgf_file, 'w') as f:
        node_ids = {node: idx + 1 for idx, node in enumerate(graph.nodes())}

        # Write nodes
        for node, node_id in node_ids.items():
            f.write(f"{node_id} {node}\n")

        # Write delimiter
        f.write("#\n")

        # Write edges
        for node1, node2, data in graph.edges(data=True):
            source = node_ids[node1]
            target = node_ids[node2]
            edge_color = get_edge_color(data['graphs'])
            f.write(f"{source} {target} {edge_color}\n")

# Read graphs from files
edges1 = read_adjacency_list('ims_join_direct_0.15/ims_join_direct.tgf')
edges2 = read_adjacency_list('ims_knowledgeretrieval_direct_0.15/ims_knowledgeretrieval_direct.tgf')
edges3 = read_adjacency_list('ims_hybridretrieval_direct_0.15/ims_hybridretrieval_direct.tgf')

# Create a unified graph
G = nx.Graph()

# Add edges from each graph to the unified graph
add_edges_to_graph(G, edges1, 'J')
add_edges_to_graph(G, edges2, 'K')
add_edges_to_graph(G, edges3, 'KJ')

# Write the unified graph to a TGF file
write_tgf(G, 'hybridretrieval/combined_tgf/combined_graph_direct_color_labels.tgf')
