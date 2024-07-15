import networkx as nx
from networkx.drawing.nx_agraph import read_dot
import os 

os.chdir('/home/iustin/Mech-Interp/Automatic-Circuit-Discovery/acdc/hybridretrieval/acdc_results/test/')

def convert_gv_to_tgf(gv_file, tgf_file):
    # Read the DOT file using NetworkX
    graph = read_dot(gv_file)

    # Open the TGF file for writing
    with open(tgf_file, 'w') as f:
        # Write nodes with their labels
        for node in graph.nodes(data=True):
            f.write(f"{node[0]} {node[0]} {node[1].get('label', '')}\n")
       
        # Write a delimiter for edges
        f.write("#\n")

        # Write edges
        for edge in graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

convert_gv_to_tgf('ims_hybridretrieval_indirect_0.15/img_new_68.gv', 'ims_hybridretrieval_indirect_0.15/ims_hybridretrieval_indirect.tgf')
convert_gv_to_tgf('ims_knowledgeretrieval_indirect_0.15/img_new_21.gv', 'ims_knowledgeretrieval_indirect_0.15/ims_knowledgeretrieval_indirect.tgf')
convert_gv_to_tgf('ims_join_indirect_0.15/img_new_40.gv', 'ims_join_indirect_0.15/ims_join_indirect.tgf')

