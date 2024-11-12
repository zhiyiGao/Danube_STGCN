import copy
import numpy as np
import os
import random
import torch
import torch.nn as nn

from dataset import LamaHDataset
from models import MLP, GCN, ResGCN, GCNII, ResGAT
from torch.nn.functional import mse_loss
from torch.utils.data import random_split
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_undirected, to_torch_coo_tensor

from torchinfo import summary
from tqdm import tqdm

from model import models

def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_edge_weights(adjacency_type, edge_attr):
    if adjacency_type == "isolated":
        return torch.zeros(edge_attr.size(0))
    elif adjacency_type == "binary":
        return torch.ones(edge_attr.size(0)) # edge_attr 张量在第 0 维（即行数）的大小。edge_attr size(358,3)
    elif adjacency_type == "stream_length":
        return edge_attr[:, 0]
    elif adjacency_type == "elevation_difference":
        return edge_attr[:, 1]
    elif adjacency_type == "average_slope":
        return edge_attr[:, 2]
    elif adjacency_type == "learned":
        return nn.Parameter(torch.nn.init.uniform_(torch.empty(edge_attr.size(0)), 0.9, 1.1))
    elif adjacency_type == "all":
        return edge_attr[:, :]
    elif adjacency_type == "all_learned":
        n = edge_attr.size(0)  # n = 9
        target_size = n * (n + 1)  # 90
        return nn.Parameter(torch.nn.init.uniform_(torch.empty(target_size), 0.9, 1.1)) # 可学习参数代码有问题
        # return nn.Parameter(torch.nn.init.uniform_(torch.empty((edge_attr+1).size(0)), 0.9, 1.1))  # 可学习参数代码有问题
    else:
        raise ValueError("invalid adjacency type", adjacency_type)


def construct_model(hparams, dataset):
    # [357]全1 learned [357](0.9-1.1)
    edge_weights = get_edge_weights(hparams["model"]["adjacency_type"], dataset.edge_attr)
    model_arch = hparams["model"]["architecture"]
    if model_arch == "MLP":
        return MLP(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"])
    elif model_arch == "GCN":
        return GCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"],
                   edge_orientation=hparams["model"]["edge_orientation"],
                   edge_weights=edge_weights
                   )
    elif model_arch == "ResGCN":
        return ResGCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                      hidden_channels=hparams["model"]["hidden_channels"],
                      num_hidden=hparams["model"]["num_layers"],
                      param_sharing=hparams["model"]["param_sharing"],
                      edge_orientation=hparams["model"]["edge_orientation"],
                      edge_weights=edge_weights)
    elif model_arch == "GCNII":
        return GCNII(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                     hidden_channels=hparams["model"]["hidden_channels"],
                     num_hidden=hparams["model"]["num_layers"],
                     param_sharing=hparams["model"]["param_sharing"],
                     edge_orientation=hparams["model"]["edge_orientation"],
                     edge_weights=edge_weights)
    elif model_arch == "ResGAT":
        return ResGAT(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                      hidden_channels=hparams["model"]["hidden_channels"],
                      num_hidden=hparams["model"]["num_layers"],
                      param_sharing=hparams["model"]["param_sharing"],
                      edge_orientation=hparams["model"]["edge_orientation"],
                      edge_weights=edge_weights)


    raise ValueError("unknown model architecture", model_arch)

def construct_STGCN_model(hparams, dataset, blocks):

    model_arch = hparams["model"]["architecture"]
    if model_arch == "stgcn":
        return models.STGCNChebGraphConv(hparams, blocks, n_vertex=len(dataset.rev_index), edge_adj=dataset.adj_matrix)  #

    raise ValueError("unknown model architecture", model_arch)
def load_dataset(path, hparams, split): # 'G:/dataset/danube/dataset_full'
    if split == "train":
        years = hparams["training"]["train_years"]
    elif split == "test":
        years = [2016, 2017]
    else:
        raise ValueError("unknown split", split)
    return LamaHDataset(path,
                        years=years,
                        root_gauge_id=hparams["data"]["root_gauge_id"],
                        rewire_graph=hparams["data"]["rewire_graph"],
                        window_size=hparams["data"]["window_size"],
                        stride_length=hparams["data"]["stride_length"],
                        lead_time=hparams["data"]["lead_time"],
                        normalized=hparams["data"]["normalized"])


def load_model_and_dataset(chkpt, dataset_path):
    model_params = chkpt["history"]["best_model_params"]
    dataset = load_dataset(dataset_path, chkpt["hparams"], split="test")
    model = construct_model(chkpt["hparams"], dataset)
    model.load_state_dict(model_params, strict=False)
    return model, dataset


def load_model_and_dataset_stgcn(chkpt, dataset_path, blocks):
    model_params = chkpt["history"]["best_model_params"]
    dataset = load_dataset(dataset_path, chkpt["hparams"], split="test")
    model = construct_STGCN_model(chkpt["hparams"], dataset, blocks)
    model.load_state_dict(model_params, strict=False)
    return model, dataset
def load_model(chkpt, dataset):
    model_params = chkpt["history"]["best_model_params"]
    model = construct_model(chkpt["hparams"], dataset)
    model.load_state_dict(model_params, strict=True)
    return model

def train_step(model, train_loader, criterion, optimizer, device, reset_running_loss_after=10):
    model.train()
    train_loss = 0.0
    running_loss = 0.0
    running_counter = 1
    # DataBatch(x=[22912, 24, 5], edge_index=[2, 22848], edge_attr=[22848, 3], y=[22912, 1], batch=[22912], ptr=[65])
    with tqdm(train_loader, desc="Training") as pbar:
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x).view(len(batch.x), -1)
            loss = criterion(pred, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs / len(train_loader.dataset)
            running_loss += loss.item() / reset_running_loss_after
            running_counter += 1
            if running_counter >= reset_running_loss_after:
                pbar.set_postfix({"loss": running_loss})
                running_counter = 1
                running_loss = 0.0
    return train_loss


def val_step(model, val_loader, criterion, device, reset_running_loss_after=10):
    model.eval()
    val_loss = 0.0
    running_loss = 0.0
    running_counter = 1
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating") as pbar:
            for batch in pbar:
                batch = batch.to(device)
                pred = model(batch.x).view(len(batch.x), -1)
                loss = criterion(pred, batch)
                val_loss += loss.item() * batch.num_graphs / len(val_loader.dataset)
                running_loss += loss.item() / reset_running_loss_after
                running_counter += 1
                if running_counter >= reset_running_loss_after:
                    pbar.set_postfix({"loss": running_loss})
                    running_counter = 1
                    running_loss = 0.0
    return val_loss


def interestingness_score(batch, dataset, device):
    mean = dataset.mean[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    std = dataset.std[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    unnormalized_discharge = mean + std * batch.x[:, :, 0]
    assert unnormalized_discharge.min() >= 0.0
    comparable_discharge = unnormalized_discharge / mean

    mean_central_diff = torch.gradient(comparable_discharge, dim=-1)[0].mean()
    trapezoid_integral = torch.trapezoid(comparable_discharge, dim=-1)

    score = 1e3 * (mean_central_diff ** 2) * trapezoid_integral
    assert not trapezoid_integral.isinf().any()
    assert not trapezoid_integral.isnan().any()
    return score.unsqueeze(-1)


def interestingness_score_normalization_const(loader, device):
    total_score = 0.0
    for batch in tqdm(loader, desc="Summing all scores"):
        total_score += interestingness_score(batch, loader.dataset, device).item()
    return total_score


def train(model, dataset, hparams):
    print(summary(model, depth=2))

    holdout_size = hparams["training"]["holdout_size"] # 拆分训练集0.8 验证集0.2
    train_dataset, val_dataset = random_split(dataset, [1 - holdout_size, holdout_size])
    train_loader = DataLoader(train_dataset, batch_size=hparams["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = lambda pred, batch: (interestingness_score(batch, dataset, device) * mse_loss(pred, batch.y, reduction="none")).mean()  # mse_loss(pred, batch.y)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams["training"]["learning_rate"],
                                 weight_decay=hparams["training"]["weight_decay"])
    model = model.to(device)
    print("Training on", device)

    history = {"train_loss": [], "val_loss": [], "best_model_params": None}

    min_val_loss = float("inf") # 初始化为正无穷
    for epoch in range(hparams["training"]["num_epochs"]):
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        val_loss = val_step(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print("[Epoch {0}/{1}] Train: {2:.4f} | Val {3:.4f}".format(
            epoch + 1, hparams["training"]["num_epochs"], train_loss, val_loss
        ))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            history["best_model_params"] = copy.deepcopy(model.state_dict())

    return history

def test(model, dataset, hparams):
    print(summary(model, depth=2))

    holdout_size = hparams["training"]["holdout_size"] # 拆分训练集0.8 验证集0.2
    train_dataset, val_dataset = random_split(dataset, [1 - holdout_size, holdout_size])
    train_loader = DataLoader(train_dataset, batch_size=hparams["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = lambda pred, batch: (interestingness_score(batch, dataset, device) * mse_loss(pred, batch.y, reduction="none")).mean()  # mse_loss(pred, batch.y)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams["training"]["learning_rate"],
                                 weight_decay=hparams["training"]["weight_decay"])
    model = model.to(device)
    print("Training on", device)

    history = {"train_loss": [], "val_loss": [], "best_model_params": None}

    min_val_loss = float("inf") # 初始化为正无穷
    for epoch in range(hparams["training"]["num_epochs"]):
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        val_loss = val_step(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print("[Epoch {0}/{1}] Train: {2:.4f} | Val {3:.4f}".format(
            epoch + 1, hparams["training"]["num_epochs"], train_loss, val_loss
        ))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            history["best_model_params"] = copy.deepcopy(model.state_dict())

    return history
def save_checkpoint(history, hparams, filename, directory="./runs"):
    directory = directory.rstrip("/")
    os.makedirs(directory, exist_ok=True)
    out_path = f"{directory}/{filename}"
    torch.save({
        "history": history,
        "hparams": hparams
    }, out_path)
    print("Saved checkpoint", out_path)


def load_checkpoint(chkpt_path):
    return torch.load(chkpt_path, map_location=torch.device("cpu"))


def evaluate_nse(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mean = dataset.mean[:, [0]].to(device)
    std_squared = dataset.std[:, [0]].square().to(device)

    with torch.no_grad():
        weighted_model_error = torch.zeros(dataset[0].num_nodes, 1).to(device)
        weighted_mean_error = torch.zeros(dataset[0].num_nodes, 1).to(device)
        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            pred = model(data.x).view(len(data.x), -1)
            model_mse = mse_loss(pred, data.y, reduction="none")
            mean_mse = mse_loss(mean, data.y, reduction="none")
            if dataset.normalized:
                model_mse *= std_squared
                mean_mse *= std_squared
            score = interestingness_score(Batch.from_data_list([data]), dataset, device)
            weighted_model_error += score * model_mse
            weighted_mean_error += score * mean_mse

    weighted_nse = 1 - weighted_model_error / weighted_mean_error
    return weighted_nse.cpu()




def evaluate_nse_acc(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mean = dataset.mean[:, [0]].to(device)
    std_squared = dataset.std[:, [0]].square().to(device)

    _m =  dataset.mean[:, [0]].to(device)
    _std = dataset.std[:, [0]].to(device)
    # 定义计数器用于计算准确率
    acc_counts = {'acc05%': 0, 'acc1%': 0, 'acc3%': 0, 'acc4%': 0, 'acc5%': 0, 'acc10%': 0}
    total_count = 0
    total_mse = 0
    total_mae = 0
    total_mape = 0

    with torch.no_grad():
        weighted_model_error = torch.zeros(dataset[0].num_nodes, 1).to(device)
        weighted_mean_error = torch.zeros(dataset[0].num_nodes, 1).to(device)

        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            pred = model(data.x).view(len(data.x), -1)

            # 反标准化还原 pred 和 data.y
            pred_unscaled = pred * _std + _m
            y_unscaled = data.y * _std + _m

            model_mse = mse_loss(pred, data.y, reduction="none")
            mean_mse = mse_loss(mean, data.y, reduction="none")
            model_mae = torch.nn.functional.l1_loss(pred, data.y, reduction="none")
            if dataset.normalized:
                model_mse *= std_squared
                mean_mse *= std_squared
                model_mae *= std_squared.sqrt()  # MAE的std平方根调整

            score = interestingness_score(Batch.from_data_list([data]), dataset, device)
            weighted_model_error += score * model_mse
            weighted_mean_error += score * mean_mse

            # 累加各项误差指标
            total_mse += model_mse.sum().item()
            total_mae += model_mae.sum().item()
            epsilon = 1e-10  # 避免除零
            total_mape += (torch.abs((pred_unscaled - y_unscaled) / (y_unscaled + epsilon)).mean() * 100).item() * len(data.y)

            # 计算预测误差与真实值之间的百分比误差
            percentage_error = torch.abs((pred_unscaled - y_unscaled) / y_unscaled + epsilon) * 100
            acc_counts['acc05%'] += torch.sum(percentage_error < 0.5).item()
            acc_counts['acc1%'] += torch.sum(percentage_error < 1).item()
            acc_counts['acc3%'] += torch.sum(percentage_error < 3).item()
            acc_counts['acc4%'] += torch.sum(percentage_error < 4).item()
            acc_counts['acc5%'] += torch.sum(percentage_error < 5).item()
            acc_counts['acc10%'] += torch.sum(percentage_error < 10).item()
            total_count += data.y.numel()  # 更新总样本数

    weighted_nse = 1 - weighted_model_error / weighted_mean_error

    avg_mse = total_mse / total_count
    rmse = torch.sqrt(torch.tensor(avg_mse))
    mae = total_mae / total_count
    mape = total_mape / total_count

    # 计算准确率
    acc0_5 = acc_counts['acc05%'] / total_count * 100
    acc1 = acc_counts['acc1%'] / total_count * 100
    acc3 = acc_counts['acc3%'] / total_count * 100
    acc4 = acc_counts['acc4%'] / total_count * 100
    acc5 = acc_counts['acc5%'] / total_count * 100
    acc10 = acc_counts['acc10%'] / total_count * 100

    return weighted_nse.cpu(), acc0_5, acc1, acc3, acc4, acc5, acc10, avg_mse, rmse, mae, mape

def calculate_predictions_and_deviations_on_gauge(model, dataset, gauge_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []
    deviations = []
    with torch.no_grad():
        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            pred = model(data.x, data.edge_index)[gauge_index]
            target = data.y[gauge_index]
            predictions.append(pred.item())
            deviations.append(abs(pred - target).item())
    return predictions, deviations


def dirichlet_energy(x, edge_index, edge_weight, normalization=None):
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization=normalization)
    lap = to_torch_coo_tensor(edge_index=edge_index, edge_attr=edge_weight)
    return 0.5 * torch.trace(torch.mm(x.T, torch.sparse.mm(lap, x)))


def evaluate_dirichlet_energy(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    dirichlet_stats = []
    with torch.no_grad():
        edge_weights = model.edge_weights.detach().nan_to_num().to(device)
        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            _, evo = model(data.x, data.edge_index, evo_tracking=True)
            dir_energies = torch.tensor([dirichlet_energy(h, data.edge_index, edge_weights) for h in evo])
            dirichlet_stats.append(dir_energies)
    dirichlet_stats = torch.stack(dirichlet_stats)
    return dirichlet_stats
