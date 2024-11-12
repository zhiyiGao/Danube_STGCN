
hparams = {
    "data": {
        "root_gauge_id": 399,
        "rewire_graph": True,
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 1,
        "normalized": True,
    },
    "model": {
        "architecture": None,  # set below
        "num_layers": None,  # set below
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": None,  # set below
        "adjacency_type": None,  # set below
    },
    "training": {
        "num_epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "random_seed": 42,
        "train_years": None,  # set below
        "holdout_size": 1/5,
    },
    "stgcn": {
        "n_vertex": None,  # set below
        "kt": 3,
        "ks": 3,
        "stblock_num": 2,
        "enable_bias": True,
        "droprate": 0.5
    }
}


DATASET_PATH = "G:/dataset/danube/dataset_full"
# Train save dir
CHECKPOINT_PATH_TRAIN = "G:/dataset/checkpoints/full/daily_stgcn_0815_learned_bs64_W24L1"

# Test load dir
CHECKPOINT_PATH_TEST = "G:/dataset/checkpoints/full/daily_stgcn_0815_learned_bs64_W24L1"

# Test save csv
OUT_FILE = "G:/dataset/checkpoints/results/daily_stgcn/daily_stgcn_0815_learned_bs64_W24L1.csv"