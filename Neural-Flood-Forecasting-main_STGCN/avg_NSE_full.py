# -*- coding: utf-8 -*-

import pandas as pd
from IPython.display import display  # Import display from IPython

results = pd.read_csv("G:/dataset/checkpoints/results/hourly_stgcn/hourly_stgcn_0815_learned_bs64_W24L1_2.csv", index_col=0, encoding='utf-8')

# def extract_results(architecture, edge_orientation, adjacency_type):
#     return results.loc[[f"{architecture}_{edge_orientation}_{adjacency_type}_{fold}" for fold in range(3)]]

def extract_results(architecture, edge_orientation, adjacency_type):
    return results.loc[[f"{architecture}_{edge_orientation}_{adjacency_type}_{1110}" for fold in range(1)]]
def print_table(architecture):
    display(pd.DataFrame(
        [
            sum([["{:.2f}%".format(100 * extract_results(architecture, edge_orientation, adjacency_type).mean(1).mean())
                    + " ± {:.2f}%".format(100 * extract_results(architecture, edge_orientation, adjacency_type).mean(1).std())]
             for edge_orientation in ["downstream"]], [])
           for adjacency_type in ["learned"]
        ], 
        columns = ["downstream (NSE)"],
        index = ["learned"]
    ))

print_table("stgcn")