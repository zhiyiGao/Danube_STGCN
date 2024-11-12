import functions

hparams = {
    "data": {
        "root_gauge_id": None,
        "rewire_graph": True,
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 6,
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
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "random_seed": 42,
        "train_years": None,  # set below
        "holdout_size": 1/5,
    },
    "stgcn":{
        "n_vertex": None, #set below
        "kt": 3,
        "ks": 3,
        "stblock_num": 2,
        "enable_bias": True,
        "droprate": 0.5
    }
}

DATASET_PATH = "G:/dataset/danube/dataset_full"
CHECKPOINT_PATH = "G:/dataset/checkpoints/subnetwork/stgcn29"


Ko = hparams["data"]["window_size"] - (hparams["stgcn"]["kt"] - 1) * 2 * hparams["stgcn"]["stblock_num"] # Ko16

# blocks: settings of channel size in st_conv_blocks and output layer,
# using the bottleneck design in st_conv_blocks
def get_blocks():
    blocks = []
    blocks.append([5]) #  [[5], [64, 16, 64], [64, 16, 64], [128, 128],[1]]
    for l in range(hparams["stgcn"]["stblock_num"]):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    return blocks


for fold_id, (train_years, test_years) in enumerate([(list(range(2013, 2016, 1)), [2016, 2017])]):
    for root_gauge_id in [29]:
        for architecture in ["stgcn"]:
            for edge_orientation in ["downstream"]:
                for adjacency_type in ["learned"]:

                    hparams["data"]["root_gauge_id"] = root_gauge_id
                    hparams["training"]["train_years"] = train_years
                    dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")

                    hparams["model"]["architecture"] = architecture
                    hparams["model"]["edge_orientation"] = edge_orientation
                    hparams["model"]["adjacency_type"] = adjacency_type
                    hparams["model"]["num_layers"] = 3

                    functions.ensure_reproducibility(hparams["training"]["random_seed"])

                    print(hparams["model"]["num_layers"], "layers used")

                    model = functions.construct_STGCN_model(hparams, dataset, get_blocks())

                    history = functions.train(model, dataset, hparams)

                    chkpt_name = f"{root_gauge_id}_{architecture}_{edge_orientation}_{adjacency_type}_{fold_id}.run"
                    functions.save_checkpoint(history, hparams, chkpt_name, directory=CHECKPOINT_PATH)