# -*- coding: utf-8 -*-

import pandas as pd

# Load the CSV file with UTF-8 encoding
results = pd.read_csv("/home/dalhxwlyjsuo/criait_gaozy/Neural-Flood-Forecasting-main/Neural-Flood-Forecasting-main/checpoints_gzy/subnetwork/512/results_532.csv", index_col=0, encoding='utf-8')

def extract_results(root_gauge_id, architecture, edge_orientation, adjacency_type):
    # Return specific rows from the DataFrame
    return results.loc[[f"/home/dalhxwlyjsuo/criait_gaozy/Neural-Flood-Forecasting-main/Neural-Flood-Forecasting-main/checpoints_gzy/subnetwork/512/{root_gauge_id}_{architecture}_{edge_orientation}_{adjacency_type}_{fold}" for fold in range(3)]]

def print_table(root_gauge_id, architecture):
    # Display a formatted table with mean and standard deviation
    display(pd.DataFrame(
        [
            sum([["{:.2f}%".format(100 * extract_results(root_gauge_id, architecture, edge_orientation, adjacency_type).mean(1).mean())
                    + " ± {:.2f}%".format(100 * extract_results(root_gauge_id, architecture, edge_orientation, adjacency_type).mean(1).std())]
             for edge_orientation in ["downstream"]], [])
           for adjacency_type in ["binary"]
        ], 
        columns=["downstream (NSE)"],
        index=["binary"]
    ))

# Example function call
print_table(211, "GCNII")  # simplest non-trivial graph (A -> B)
