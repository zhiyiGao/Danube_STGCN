# -*- coding: utf-8 -*-

import pandas as pd
from IPython.display import display  # Import display from IPython

# Load the CSV file with UTF-8 encoding
results = pd.read_csv("G:/dataset/checkpoints/results/stgcn/results_stgcn_29.csv", index_col=0, encoding='utf-8')

# def extract_results(root_gauge_id, architecture, edge_orientation, adjacency_type):
#     # Return specific rows from the DataFrame
#     return results.loc[[f"{root_gauge_id}_{architecture}_{edge_orientation}_{adjacency_type}_{fold}" for fold in range(3)]]

def extract_results(root_gauge_id, architecture, edge_orientation, adjacency_type):
    # Return specific rows from the DataFrame
    return results.loc[[f"{root_gauge_id}_{architecture}_{edge_orientation}_{adjacency_type}_{fold}" for fold in range(1)]]
def print_table(root_gauge_id, architecture):
    # Display a formatted table with mean and standard deviation
    display(pd.DataFrame(
        [
            sum([["{:.2f}%".format(100 * extract_results(root_gauge_id, architecture, edge_orientation, adjacency_type).mean(1).mean())
                    + " ± {:.2f}%".format(100 * extract_results(root_gauge_id, architecture, edge_orientation, adjacency_type).mean(1).std())]
             for edge_orientation in ["downstream"]], [])
           for adjacency_type in ["learned"]
        ], 
        columns=["downstream (NSE)"],
        index=["learned"]
    ))

# Example function call
print_table(29, "stgcn")  # simplest non-trivial graph (A -> B)