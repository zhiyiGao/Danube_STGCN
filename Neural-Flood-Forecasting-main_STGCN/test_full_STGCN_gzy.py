import functions
import pandas as pd
import config
from model.models import get_blocks

# 使用配置中的参数
hparams = config.hparams
DATASET_PATH = config.DATASET_PATH
CHECKPOINT_PATH = config.CHECKPOINT_PATH_TEST

OUT_FILE = config.OUT_FILE

results_df = pd.DataFrame()


# add_acc
for architecture in ["stgcn"]:
    for edge_orientation in ["downstream"]:
        for adjacency_type in ["learned"]:
            run_id = f"{architecture}_{edge_orientation}_{adjacency_type}_{0}"
            print(run_id)
            chkpt = functions.load_checkpoint(f"{CHECKPOINT_PATH}/{run_id}.run")
            model, dataset = functions.load_model_and_dataset_stgcn(chkpt, DATASET_PATH, get_blocks(hparams))
            test_nse, acc0_5, acc1, acc3, acc4, acc5, acc10, avg_mse, rmse, mae, mape = functions.evaluate_nse_acc(model, dataset)

            # 打印acc1, acc3, acc5, acc10的值
            print(f"Results for {run_id}:")
            print(f"  acc05%: {acc0_5:.3f}%")
            print(f"  acc1%: {acc1:.3f}%")
            print(f"  acc3%: {acc3:.3f}%")
            print(f"  acc4%: {acc4:.3f}%")
            print(f"  acc5%: {acc5:.3f}%")
            print(f"  acc10%: {acc10:.3f}%")

            print(f"MSE: {avg_mse:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"MAE: {mae:.3f}")
            print(f"MAPE: {mape:.3f}")

            results_df.loc[run_id, range(len(test_nse))] = test_nse.squeeze().numpy()
            results_df.to_csv(OUT_FILE)