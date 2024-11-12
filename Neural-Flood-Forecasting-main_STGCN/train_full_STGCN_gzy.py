import functions
from model.models import get_blocks
import config


# 使用配置中的参数
hparams = config.hparams
DATASET_PATH = config.DATASET_PATH
CHECKPOINT_PATH = config.CHECKPOINT_PATH_TRAIN


# blocks: settings of channel size in st_conv_blocks and output layer,
# using the bottleneck design in st_conv_blocks

for fold_id, (train_years, test_years) in enumerate([(list(range(2008, 2016, 1)), [2016, 2017])]):
    for architecture in ["stgcn"]:
        for edge_orientation in ["downstream"]:
            for adjacency_type in ["learned"]:
                hparams["training"]["train_years"] = train_years
                dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")

                hparams["model"]["architecture"] = architecture
                hparams["model"]["edge_orientation"] = edge_orientation
                hparams["model"]["adjacency_type"] = adjacency_type
                hparams["model"]["num_layers"] = 3

                functions.ensure_reproducibility(hparams["training"]["random_seed"])

                print(hparams["model"]["num_layers"], "layers used")
                model = functions.construct_STGCN_model(hparams, dataset, get_blocks(hparams))

                history = functions.train(model, dataset, hparams)

                chkpt_name = f"{architecture}_{edge_orientation}_{adjacency_type}_{fold_id}.run"
                functions.save_checkpoint(history, hparams, chkpt_name, directory=CHECKPOINT_PATH)