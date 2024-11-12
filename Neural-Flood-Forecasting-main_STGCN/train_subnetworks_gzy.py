import functions

hparams = {
    "data": {
        "root_gauge_id": None,  # set below,
        "rewire_graph": False,
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 6,
        "normalized": True,
    },
    "model": {
        "architecture": None,  # set below
        "num_layers": None,  # set below
        "hidden_channels": 512,
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
    }
}


DATASET_PATH = "G:/dataset/danube/dataset_full"
CHECKPOINT_PATH = "G:/dataset/checkpoints/subnetwork/GCNII211"

for fold_id, (train_years, test_years) in enumerate([(list(range(2008, 2016, 1)), [2016, 2017])]):
    for root_gauge_id in [211]:
        for architecture in ["GCNII"]:
            for edge_orientation in ["downstream"]: ##这里默认让他不操作，写成下游
                for adjacency_type in ["all_learned"]:
                    hparams["data"]["root_gauge_id"] = root_gauge_id
                    hparams["training"]["train_years"] = train_years
                    dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")

                    hparams["model"]["architecture"] = architecture
                    hparams["model"]["edge_orientation"] = edge_orientation
                    hparams["model"]["adjacency_type"] = adjacency_type
                    hparams["model"]["num_layers"] = 4
                    functions.ensure_reproducibility(hparams["training"]["random_seed"])
                    print(hparams["model"]["num_layers"], "layers used")
                    model = functions.construct_model(hparams, dataset)
                    history = functions.train(model, dataset, hparams)

                    chkpt_name = f"{root_gauge_id}_{architecture}_{edge_orientation}_{adjacency_type}_{fold_id}.run"
                    functions.save_checkpoint(history, hparams, chkpt_name, directory=CHECKPOINT_PATH)
