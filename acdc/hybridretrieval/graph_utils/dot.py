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

        # Track node memberships
        if 'graphs' not in graph.nodes[node1]:
            graph.nodes[node1]['graphs'] = set()
        graph.nodes[node1]['graphs'].add(graph_label)

        if 'graphs' not in graph.nodes[node2]:
            graph.nodes[node2]['graphs'] = set()
        graph.nodes[node2]['graphs'].add(graph_label)

def get_color(graphs):
    color_map = {
        frozenset({'J'}): 'red',
        frozenset({'K'}): 'yellow',
        frozenset({'M'}): 'blue',
        frozenset({'J', 'K'}): 'orange',
        frozenset({'J', 'M'}): 'purple',
        frozenset({'K', 'M'}): 'green',
        frozenset({'J', 'K', 'M'}): 'cyan'
    }
    return color_map.get(frozenset(graphs), 'grey')

def get_label(graphs):
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
        f.write("digraph MyGraph {\n")
        node_ids = {node: idx + 1 for idx, node in enumerate(graph.nodes())}
        # Write nodes
        for node, node_id in node_ids.items():
            node_color = get_color(graph.nodes[node]['graphs'])
            f.write(f'{node} [label="{node}", shape=ellipse, color="{node_color}"];\n')

        # Write edges
        for node1, node2, data in graph.edges(data=True):
            source = node1
            target = node2
            edge_color = get_color(data['graphs'])
            edge_label = get_label(data['graphs'])
            f.write(f'{source} -> {target} [label="{edge_label}", color="{edge_color}"];\n')
        
        f.write("}\n")

# Read graphs from files
edges1 = read_adjacency_list('hybridretrieval/acdc_results/ims_hybridretrieval_indirect_0.15/ims_hybridretrieval_indirect.tgf')
edges2 = read_adjacency_list('hybridretrieval/acdc_results/ims_join_indirect_0.15/ims_join_indirect_0.15.tgf')
edges3 = read_adjacency_list('hybridretrieval/acdc_results/ims_knowledgeretrieval_indirect_0.15/ims_knowledgeretrieval_indirect_0.15.tgf')

# Create a unified graph
G = nx.DiGraph()

# Add edges from each graph to the unified graph
add_edges_to_graph(G, edges1, 'J')
add_edges_to_graph(G, edges2, 'K')
add_edges_to_graph(G, edges3, 'M')

# Write the unified graph as a gv (dot) file
write_tgf(G, 'hybridretrieval/combined_tgf/combined_kbicr_dot.gv')