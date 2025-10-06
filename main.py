# ==============================================================================
#
# Filename: main.py
#
# ==============================================================================

import argparse
import os
import time
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorcircuit as tc
from scipy.stats import spearmanr, kendalltau, rankdata
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# --- 1. Utility Functions and Configuration Imports ---
from utils.common import ensure_dir, load_pkl, save_pkl, format_execution_time
from utils import graph_utils as utils_graph
from utils.plot.plot import (
    plot_loss_curves,
    plot_spearman_curves,
    plot_ndcg_curves,
    plot_metric_curves, scatter_plot_basic, plot_r2_curves, plot_mae_curves
)
from utils.log_experiment import log_to_excel
from utils.metrics.ranking_metrics import spearman_corr, NDCG, get_top_k_energy

from config import TASK_CONFIGS
from model.DAGTransformer import DAGTransformer
from model.MLP import MLP, SqueezeOutput
from model.losses.SoftNDCGLoss import SoftNDCGLoss
from quantum_gates import Gate

import matplotlib.pyplot as plt


# --- Matplotlib Chinese Display Configuration ---
def configure_matplotlib_for_chinese():
    """Configures Matplotlib to support Chinese display (optional)."""
    try:
        # This part is for displaying Chinese characters.
        # For English-only environments, this can be safely removed.
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' is a common Chinese font
        plt.rcParams['axes.unicode_minus'] = False
        print("Matplotlib configured to use 'SimHei' font for Chinese characters.")
    except Exception as e:
        print(f"Warning: Failed to configure Chinese font 'SimHei': {e}")
        print("Please ensure 'SimHei' or another Chinese font is installed on your system if you need Chinese support in plots.")


# Commenting out the call as it's not needed for an English repository.
# You can uncomment it if you need to generate plots with Chinese labels.
# configure_matplotlib_for_chinese()

# --- 3. Global Configuration ---
tc.set_dtype("complex128")
tc.set_backend("pytorch")


# ==============================================================================
#                          Helper and Utility Functions
# ==============================================================================

def pad_adj_ops(adj_list, ops_list, max_nodes):
    """Pads adjacency matrices and operation features to the maximum number of nodes in the dataset."""
    padded_adj_list, padded_ops_list = [], []
    for adj, ops in zip(adj_list, ops_list):
        num_nodes = adj.shape[0]
        padded_adj = np.zeros((max_nodes, max_nodes))
        padded_adj[:num_nodes, :num_nodes] = adj
        padded_ops = np.zeros((max_nodes, ops.shape[1]))
        padded_ops[:num_nodes, :] = ops
        padded_adj_list.append(padded_adj)
        padded_ops_list.append(padded_ops)
    return np.stack(padded_adj_list), np.stack(padded_ops_list)


# [Normalization Change] Label normalization helper functions
def scale_labels_standard_minmax(labels):
    """
    Applies standard Min-Max normalization to labels, scaling all values to the [0, 1] range.
    """
    labels = np.array(labels)
    min_val = np.min(labels)
    max_val = np.max(labels)

    if max_val == min_val:
        return np.full_like(labels, 0.5, dtype=float)

    scaled_labels = (labels - min_val) / (max_val - min_val)
    return scaled_labels


def scale_labels_quantile(labels):
    """
    Uses quantile transformation to map labels to a uniform distribution in [0, 1] (without sklearn).
    """
    labels = np.array(labels)
    n = len(labels)
    if n <= 1:
        return np.full(n, 0.5)

    ranks = rankdata(labels, method='average') - 1

    # Normalize ranks to the [0, 1] interval
    scaled_labels = ranks / (n - 1)
    return scaled_labels


def _process_single_pretrain_item(args_tuple):
    """Processes a single pre-training data item for parallelization."""
    adj_orig, ops_padded, adj_padded, label, max_nodes = args_tuple
    num_real_nodes = adj_orig.shape[0]

    edge_index = utils_graph.adj2edge_index(adj_orig)
    temp_data = Data(edge_index=edge_index, num_nodes=num_real_nodes)
    real_depths = utils_graph.get_node_depths(temp_data)
    padded_depths = torch.zeros(max_nodes, dtype=torch.long)
    padded_depths[:num_real_nodes] = real_depths
    mask = torch.zeros(max_nodes, dtype=torch.bool)
    mask[:num_real_nodes] = True
    reachability_edge_index = utils_graph.get_reachability_edge_index(temp_data)
    data = Data(x=torch.Tensor(ops_padded), adj=torch.Tensor(adj_padded),
                edge_index=edge_index, depths=padded_depths, mask=mask, y=float(label),
                reachability_edge_index=reachability_edge_index)
    return data


def _process_single_finetune_item(args_tuple):
    """Processes a single downstream task data item for parallelization."""
    adj_orig, ops_padded, adj_padded, label_raw, max_nodes = args_tuple
    num_real_nodes = adj_orig.shape[0]

    edge_index = utils_graph.adj2edge_index(adj_orig)
    temp_data = Data(edge_index=edge_index, num_nodes=num_real_nodes)
    real_depths = utils_graph.get_node_depths(temp_data)
    reachability_edge_index = utils_graph.get_reachability_edge_index(temp_data)
    padded_depths = torch.zeros(max_nodes, dtype=torch.long)
    padded_depths[:num_real_nodes] = real_depths
    mask = torch.zeros(max_nodes, dtype=torch.bool)
    mask[:num_real_nodes] = True
    pyg_item = Data(x=torch.Tensor(ops_padded), adj=torch.Tensor(adj_padded),
                    edge_index=edge_index, depths=padded_depths, mask=mask,
                    reachability_edge_index=reachability_edge_index, y_raw=float(label_raw))
    return pyg_item


def load_pretrain_summary(pretrained_model_path):
    """Loads the pre-training summary file (pretrain_summary.json) from the pretrained model's directory."""
    if not pretrained_model_path or not os.path.exists(pretrained_model_path):
        return {}

    summary_path = os.path.join(os.path.dirname(pretrained_model_path), "pretrain_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            print(f"Successfully loaded pre-training summary from {summary_path}.")
            return summary_data
        except Exception as e:
            print(f"Warning: Failed to load pre-training summary: {e}")
            return {}
    else:
        print(f"Warning: Pre-training summary file not found: {summary_path}")
        return {}


# ==============================================================================
#                                Model Definitions
# ==============================================================================

class PretrainModel(torch.nn.Module):
    """The complete model for pre-training (Encoder + MLP head)."""

    def __init__(self, encoder, mlp_head):
        super().__init__()
        self.encoder = encoder
        self.mlp_head = mlp_head
        self.sigmoid = torch.nn.Sigmoid()
        self.squeeze = SqueezeOutput()

    def forward(self, data):
        graph_embedding = self.encoder(data)
        output = self.mlp_head(graph_embedding)
        return self.squeeze(self.sigmoid(output))

    def get_encoder(self):
        return self.encoder


class DownstreamModel(torch.nn.Module):
    """The complete model for downstream tasks, including logic to freeze/unfreeze the encoder."""

    def __init__(self, encoder, mlp_head, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.mlp_head = mlp_head
        self.sigmoid = torch.nn.Sigmoid()
        self.squeeze = SqueezeOutput()
        self.is_encoder_frozen = freeze_encoder
        if self.is_encoder_frozen:
            self.freeze_encoder_params()

    def forward(self, data):
        if self.is_encoder_frozen:
            self.encoder.eval()
            with torch.no_grad():
                graph_embedding = self.encoder(data)
        else:
            self.encoder.train()
            graph_embedding = self.encoder(data)

        prediction = self.mlp_head(graph_embedding)
        return self.squeeze(self.sigmoid(prediction))

    def freeze_encoder_params(self):
        """Freezes all parameters of the encoder."""
        print("--- Freezing encoder parameters ---")
        self.is_encoder_frozen = True
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(
            f"--- Encoder parameters frozen. Current trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,} ---")

    def unfreeze_encoder_params(self):
        """Unfreezes all parameters of the encoder."""
        print("--- Unfreezing encoder parameters ---")
        self.is_encoder_frozen = False
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(
            f"--- Encoder parameters unfrozen. Current trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,} ---")


# ==============================================================================
#                            Data Loading Section
# ==============================================================================

def load_pretrain_data(args):
    """Loads and prepares data for [task-specific] pre-training."""
    print(f"--- Loading pre-training data for task '{args.task_name}' ---")
    data_path = f"data/raw/grid_16q_{args.task_name}/pretrain/run_{args.seed}/"

    # Adjust cache filename based on normalization strategy to avoid mixing data
    scaling_suffix = f"_{args.pretrain_label_scaling}" if args.pretrain_label_scaling != 'none' else ""
    pyg_data_path = os.path.join(data_path, f"pyg_data_list_samples{args.sample_lines}_masked{scaling_suffix}.pkl")

    if os.path.exists(pyg_data_path):
        print(f"Found pre-processed data, loading from: {pyg_data_path}")
        pyg_data_list = load_pkl(pyg_data_path)
    else:
        print("Pre-processed data not found, creating from raw files...")
        circuits = load_pkl(os.path.join(data_path, "adj_feat_matrix.pkl"))[:args.sample_lines]
        labels = load_pkl(os.path.join(data_path, f"RF_result_samplelines{args.sample_lines}_noise.pkl"))

        # Apply label normalization here
        if args.pretrain_label_scaling == 'standard-minmax':
            print("Applying [Standard Min-Max] normalization to pre-training labels...")
            labels = scale_labels_standard_minmax(labels)
            print(f"Normalized labels (first 5): {np.round(labels[:5], 4)}")
        elif args.pretrain_label_scaling == 'quantile':
            print("Applying [Quantile] normalization to pre-training labels...")
            labels = scale_labels_quantile(labels)
            print(f"Normalized labels (first 5): {np.round(labels[:5], 4)}")
        else:
            print("No label normalization applied for pre-training.")

        adj_list, ops_list = [c[0] for c in circuits], [c[1] for c in circuits]
        max_nodes = max(adj.shape[0] for adj in adj_list) if adj_list else 0
        adj_padded_list, ops_padded_list = pad_adj_ops(adj_list, ops_list, max_nodes)

        tasks = [(adj_list[i], ops_padded_list[i], adj_padded_list[i], labels[i], max_nodes) for i in
                 range(len(adj_list))]

        with ProcessPoolExecutor() as executor:
            results_iterator = executor.map(_process_single_pretrain_item, tasks)
            pyg_data_list = list(tqdm(results_iterator, total=len(tasks), desc="Creating PyG data objects in parallel"))

        print(f"Saving pre-processed data to: {pyg_data_path}")
        save_pkl(pyg_data_list, pyg_data_path)

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(pyg_data_list))
    rng.shuffle(indices)
    split_idx = int(len(pyg_data_list) * (1 - args.val_split))
    train_data = [pyg_data_list[i] for i in indices[:split_idx]]
    val_data = [pyg_data_list[i] for i in indices[split_idx:]]
    train_loader = DataLoader(train_data, batch_size=args.pretrain_bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.pretrain_bs, shuffle=False, drop_last=True)
    return train_loader, val_loader


def load_finetune_data(args):
    """Loads and prepares data for fine-tuning."""
    print(f"--- Loading and preparing data for downstream task '{args.task_name}' ---")

    def get_or_create_processed_data(seed, task_name):
        data_path = f'data/raw/grid_16q_{task_name}/training/run_{seed}'
        cache_path = os.path.join(data_path, 'processed_dag_data.pkl')
        if os.path.exists(cache_path):
            return load_pkl(cache_path)

        print(f"Creating cached data for task '{task_name}' with seed {seed}...")
        matrix_all = load_pkl(f"{data_path}/adj_feat_matrix.pkl")
        labels_raw_all = load_pkl(f"{data_path}/energy.pkl")
        num_labeled = len(labels_raw_all)
        matrix, labels_raw = matrix_all[:num_labeled], labels_raw_all
        adj_list_orig, ops_list_orig = [c[0] for c in matrix], [c[1] for c in matrix]
        max_nodes = max(adj.shape[0] for adj in adj_list_orig) if adj_list_orig else 0
        adj_padded_list, ops_padded_list = pad_adj_ops(adj_list_orig, ops_list_orig, max_nodes)

        tasks = [
            (adj_list_orig[i], ops_padded_list[i], adj_padded_list[i], labels_raw[i], max_nodes)
            for i in range(len(matrix))
        ]

        with ProcessPoolExecutor() as executor:
            results_iterator = executor.map(_process_single_finetune_item, tasks)
            pyg_list = list(tqdm(results_iterator, total=len(tasks), desc=f"Parallel processing for seed {seed}"))

        save_pkl(pyg_list, cache_path)
        return pyg_list

    train_data_full = get_or_create_processed_data(args.seed, args.task_name)
    valid_data_full = get_or_create_processed_data((args.seed + 1) % 5, args.task_name)
    train_data = train_data_full[:args.num_train_samples]
    valid_data = valid_data_full[:args.num_val_samples]
    print(f"Using {len(train_data)} training samples and {len(valid_data)} validation samples.")

    train_labels_neg = -np.array([d.y_raw for d in train_data])
    min_val, max_val = np.min(train_labels_neg), np.max(train_labels_neg)
    range_val = max_val - min_val if max_val > min_val else 1.0
    for d in train_data: d.y = (-d.y_raw - min_val) / range_val
    for d in valid_data: d.y = (-d.y_raw - min_val) / range_val

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader


# ==============================================================================
#                            Pre-training Core Logic
# ==============================================================================

def build_pretrain_model(args):
    """Builds the model for pre-training."""
    encoder = DAGTransformer(in_dim=args.input_dim, d_model=args.hidden_dim, num_heads=args.num_heads,
                             num_layers=args.num_layers, dropout=args.dropout)
    mlp_head = MLP(input_dim=args.hidden_dim, hidden_dims=[args.hidden_dim], output_dim=1,
                   activation='relu', use_bn=True, dropout=args.dropout)
    model = PretrainModel(encoder, mlp_head)
    print("--- Pre-training model built ---")
    print(model)
    return model


def calculate_pretrain_metrics(preds, targets, loss=None):
    """Calculates various metrics for the pre-training task."""
    metrics = {'loss': loss if loss is not None else 0}
    preds, targets = np.array(preds).flatten(), np.array(targets).flatten()
    if np.std(targets) > 0 and np.std(preds) > 0:
        metrics['spearman'], _ = spearmanr(preds, targets)
        metrics['kendalltau'], _ = kendalltau(preds, targets)
        ss_total = np.sum((targets - np.mean(targets)) ** 2)
        ss_residual = np.sum((targets - preds) ** 2)
        metrics['r2'] = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    else:
        metrics['spearman'], metrics['kendalltau'], metrics['r2'] = 0, 0, 0
    metrics['mae'] = np.mean(np.abs(targets - preds))
    return metrics


def pretrain_train_epoch(model, loader, optimizer, loss_func, device):
    """Executes one epoch of pre-training."""
    model.train()
    for batch in loader:
        batch, target = batch.to(device), batch.y.float()
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()


def pretrain_evaluate(model, loader, loss_func, device):
    """Evaluates the model's performance on the pre-training task."""
    model.eval()
    total_loss, all_preds, all_targets = 0, [], []
    with torch.no_grad():
        for batch in loader:
            batch, target = batch.to(device), batch.y.float()
            pred = model(batch)
            loss = loss_func(pred, target)
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return calculate_pretrain_metrics(all_preds, all_targets, avg_loss)


def run_pretraining(args, cfg):
    """Executes the complete [task-specific] pre-training workflow."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Pre-training for task '{args.task_name}' on device {device} ---")

    base_path = "pretrain_results"
    hyperparam_str = f"pbs{args.pretrain_bs}_plr{args.lr_pretrain:.0e}_ploss-{args.loss_func_pretrain}_s{args.sample_lines}"
    exp_path = os.path.join(base_path, args.task_name, hyperparam_str, f"seed{args.seed}")

    model_save_path = os.path.join(exp_path, 'models')
    log_path = os.path.join(exp_path, 'logs')
    plot_path = os.path.join(exp_path, 'plots')
    for path in [model_save_path, log_path, plot_path]: ensure_dir(path)
    print(f"Results for this pre-training run on task '{args.task_name}' will be saved to: {exp_path}")

    train_loader, val_loader = load_pretrain_data(args)
    model = build_pretrain_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_pretrain)

    if args.loss_func_pretrain == 'mse':
        loss_func = torch.nn.MSELoss()
    elif args.loss_func_pretrain == 'huber':
        loss_func = torch.nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported pre-training loss function: {args.loss_func_pretrain}")
    print(f"Using {args.loss_func_pretrain.upper()} loss function for pre-training.")

    print("\n--- Starting Pre-training ---")
    best_val_spearman = -1.0
    history_keys = ['loss', 'spearman', 'kendalltau', 'r2', 'mae']
    metrics_history = {'train': {k: [] for k in history_keys}, 'val': {k: [] for k in history_keys}}
    final_val_metrics = {}

    for epoch in tqdm(range(args.epochs_pretrain), desc="Pre-training Progress", unit="epoch"):
        pretrain_train_epoch(model, train_loader, optimizer, loss_func, device)
        train_metrics = pretrain_evaluate(model, train_loader, loss_func, device)
        val_metrics = pretrain_evaluate(model, val_loader, loss_func, device)
        scheduler.step()

        final_val_metrics = val_metrics

        for key in history_keys:
            metrics_history['train'][key].append(train_metrics[key])
            metrics_history['val'][key].append(val_metrics[key])

        print(f"\nEpoch {epoch + 1:03d}/{args.epochs_pretrain} | "
              f"Train [Loss: {train_metrics['loss']:.4f}, Spearman: {train_metrics['spearman']:.4f}] | "
              f"Valid [Loss: {val_metrics['loss']:.4f}, Spearman: {val_metrics['spearman']:.4f}]")

        if val_metrics['spearman'] > best_val_spearman:
            best_val_spearman = val_metrics['spearman']
            torch.save(model.get_encoder().state_dict(), os.path.join(model_save_path, "best_encoder.pt"))
            print(f"  -> New best model saved (Validation Spearman: {best_val_spearman:.4f})")

    print("\n--- Pre-training Finished ---")
    last_model_path = os.path.join(model_save_path, "last_encoder.pt")
    torch.save(model.get_encoder().state_dict(), last_model_path)
    print(f"Model from the last epoch saved to: {last_model_path}")

    summary_filepath = os.path.join(model_save_path, "pretrain_summary.json")
    final_val_metrics_serializable = {k: float(v) for k, v in final_val_metrics.items()}
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_val_metrics_serializable, f, indent=4)
    print(f"Pre-training summary saved to: {summary_filepath}")

    print("\n--- Generating pre-training result plots ---")
    plot_loss_curves(metrics_history['train']['loss'], metrics_history['val']['loss'], 'Pre-training Loss Curve', plot_path,
                     'pretrain_loss_curve.png')
    plot_spearman_curves(metrics_history['train']['spearman'], metrics_history['val']['spearman'],
                         'Pre-training Spearman Correlation Curve', plot_path, 'pretrain_spearman_curve.png')
    plot_r2_curves(metrics_history['train']['r2'], metrics_history['val']['r2'], 'Pre-training R2 Score Curve', plot_path,
                     'pretrain_r2_curve.png')
    plot_mae_curves(metrics_history['train']['mae'], metrics_history['val']['mae'], 'Pre-training MAE Curve', plot_path,
                     'pretrain_mae_curve.png')
    print(f"Plots saved to: {plot_path}")
    print(f'\nTotal pre-training time: {format_execution_time(time.time() - start_time)}')


# ==============================================================================
#                      Fine-tuning and Evaluation Core Logic
# ==============================================================================
def setup_experiment_paths(args):
    """Sets up paths and loggers for the fine-tuning experiment."""
    if not args.use_pretrained:
        mode_str = "scratch"
    elif args.two_stage_finetune:
        mode_str = "2stage"
    elif args.freeze_encoder:
        mode_str = "frozen"
    else:
        mode_str = "full"

    base_path = f"finetune_results"

    if args.two_stage_finetune:
        lr_str = f"lr_h{args.lr_finetune_head:.0e}_e{args.lr_finetune_encoder:.0e}"
    else:
        lr_str = f"lr{args.lr:.0e}"
    finetune_params_str = f"ft_T{args.num_train_samples}_B{args.batch_size}_{lr_str}_L-{args.loss_func}"

    if args.use_pretrained:
        pretrain_params_str = f"pt_pbs{args.pretrain_bs}_plr{args.lr_pretrain:.0e}_ploss-{args.loss_func_pretrain}_s{args.sample_lines}"
        group_name = f"{mode_str}_{finetune_params_str}_{pretrain_params_str}"
    else:
        group_name = f"{mode_str}_{finetune_params_str}"

    run_name = f"seed{args.seed}"

    exp_path = os.path.join(base_path, args.task_name, group_name, run_name)
    paths = {
        'run_path': exp_path,
        'excel_path': os.path.join(base_path, args.task_name, group_name),
        'plot_path': os.path.join(exp_path, 'plots'),
        'log_path': os.path.join(exp_path, 'logs')
    }
    for path_key in ['run_path', 'plot_path', 'log_path']:
        ensure_dir(paths[path_key])

    writer = None

    print(f"Experiment Mode: {mode_str}")
    print(f"Results will be saved in: {os.path.join(base_path, args.task_name, group_name)}")
    print(f"Experiment Group: {group_name}\nCurrent Run: {run_name}")
    return paths, writer, group_name


def build_finetune_model(args):
    """Builds the downstream task model for fine-tuning."""
    print("--- Building downstream model (Encoder: DAGTransformer) ---")
    encoder = DAGTransformer(in_dim=args.input_dim, d_model=args.hidden_dim, num_heads=args.num_heads,
                             num_layers=args.num_layers, dropout=args.dropout)
    mlp_head = MLP(input_dim=args.hidden_dim, hidden_dims=[64, 32], output_dim=1,
                   activation='relu', use_bn=True, dropout=args.dropout_mlp)
    should_freeze = args.use_pretrained and (args.two_stage_finetune or args.freeze_encoder)
    model = DownstreamModel(encoder, mlp_head, freeze_encoder=should_freeze)
    if args.use_pretrained:
        if not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(f"Pre-trained model not found at: {args.pretrained_path}")
        print(f"Loading pre-trained weights from: {args.pretrained_path}")
        model.encoder.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'))
        print("Successfully loaded pre-trained weights.")
    else:
        print("Training from scratch. Both encoder and MLP head are trainable.")
    print("--- Downstream model built successfully ---")
    print(model)
    return model


def run_one_finetune_epoch(model, loader, optimizer, loss_func, device, is_train):
    """Runs one training or evaluation epoch for fine-tuning."""
    model.train() if is_train else model.eval()
    if is_train and not model.is_encoder_frozen:
        model.encoder.train()
    total_loss, all_preds, all_targets_raw = 0, [], []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            target_norm = batch.y.float()
            pred_norm = model(batch)
            loss = loss_func(pred_norm, target_norm)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            all_preds.extend(pred_norm.cpu().detach().numpy().flatten())
            all_targets_raw.extend(batch.y_raw.cpu().numpy().flatten())
    num_graphs_total = len(loader.dataset)
    all_targets_neg = -np.array(all_targets_raw)
    metrics = {
        'loss': total_loss / num_graphs_total if num_graphs_total > 0 else 0,
        'spearman': spearman_corr(all_preds, all_targets_neg),
        'ndcg@1': NDCG(y_pred=all_preds, y_true=all_targets_neg, ats=1),
        'ndcg@5': NDCG(y_pred=all_preds, y_true=all_targets_neg, ats=5),
        'ndcg@10': NDCG(y_pred=all_preds, y_true=all_targets_neg, ats=10),
        'ndcg@20': NDCG(y_pred=all_preds, y_true=all_targets_neg, ats=20),
        'ndcg@50': NDCG(y_pred=all_preds, y_true=all_targets_neg, ats=50),
        'ndcg@100': NDCG(y_pred=all_preds, y_true=all_targets_neg, ats=100),
        'top1': get_top_k_energy(y_pred=all_preds, y_true=all_targets_neg, k=1),
        'top5': get_top_k_energy(y_pred=all_preds, y_true=all_targets_neg, k=5),
        'top10': get_top_k_energy(y_pred=all_preds, y_true=all_targets_neg, k=10),
        'top20': get_top_k_energy(y_pred=all_preds, y_true=all_targets_neg, k=20),
        'top50': get_top_k_energy(y_pred=all_preds, y_true=all_targets_neg, k=50),
        'top100': get_top_k_energy(y_pred=all_preds, y_true=all_targets_neg, k=100),
    }
    return metrics, all_preds, all_targets_neg


def run_finetuning(args, cfg):
    """Executes the complete fine-tuning workflow, supporting multiple strategies."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Fine-tuning on device {device} ---")
    paths, writer, group_name = setup_experiment_paths(args)
    train_loader, valid_loader = load_finetune_data(args)
    model = build_finetune_model(args).to(device)
    if args.loss_func == 'mse':
        loss_func = nn.MSELoss()
    elif args.loss_func == 'SoftNDCG':
        loss_func = SoftNDCGLoss(temperature=args.temperature, k=args.k, metric_function=args.metric_function)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_func}")
    print(f"Using {args.loss_func.upper()} loss function for fine-tuning.")

    best_val_metric = -1.0
    global_epoch_counter = 0
    history_keys = ['loss', 'spearman', 'ndcg@10', 'ndcg@20', 'top1', 'top5', 'top10']
    metrics_history = {'train': {k: [] for k in history_keys}, 'val': {k: [] for k in history_keys}}

    if args.use_pretrained and args.two_stage_finetune:
        print("\n--- Starting two-stage fine-tuning ---")
        print("\n--- Stage 1: Training predictor only (Encoder frozen) ---")
        optimizer = torch.optim.AdamW(model.mlp_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_head_only)
        for epoch in tqdm(range(args.epochs_head_only), desc="Stage 1 Progress", unit="epoch"):
            run_one_finetune_epoch(model, train_loader, optimizer, loss_func, device, is_train=True)
            train_metrics, _, _ = run_one_finetune_epoch(model, train_loader, None, loss_func, device, is_train=False)
            val_metrics, _, _ = run_one_finetune_epoch(model, valid_loader, None, loss_func, device, is_train=False)
            scheduler.step()
            print(
                f"Epoch {global_epoch_counter + 1:03d} | [Stage 1] | Train Sp:{train_metrics['spearman']:.4f}, NDCG@20:{train_metrics['ndcg@20']:.4f} | Valid Sp:{val_metrics['spearman']:.4f}, NDCG@20:{val_metrics['ndcg@20']:.4f}")
            for key in history_keys:
                metrics_history['train'][key].append(train_metrics[key])
                metrics_history['val'][key].append(val_metrics[key])
            global_epoch_counter += 1

        last_stage1_model_path = os.path.join(paths['run_path'], "last_stage1_model.pt")
        torch.save(model.state_dict(), last_stage1_model_path)
        print(f"--- Stage 1 finished, model saved to: {last_stage1_model_path} ---")

        print("\n--- Stage 2: Fine-tuning the entire model (Encoder unfrozen with differential learning rates) ---")
        model.load_state_dict(torch.load(os.path.join(paths['run_path'], "last_stage1_model.pt")))
        model.unfreeze_encoder_params()
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': args.lr_finetune_encoder},
            {'params': model.mlp_head.parameters(), 'lr': args.lr_finetune_head}
        ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_full_finetune)
        for epoch in tqdm(range(args.epochs_full_finetune), desc="Stage 2 Progress", unit="epoch"):
            run_one_finetune_epoch(model, train_loader, optimizer, loss_func, device, is_train=True)
            train_metrics, _, _ = run_one_finetune_epoch(model, train_loader, None, loss_func, device, is_train=False)
            val_metrics, _, _ = run_one_finetune_epoch(model, valid_loader, None, loss_func, device, is_train=False)
            scheduler.step()
            print(
                f"Epoch {global_epoch_counter + 1:03d} | [Stage 2] | Train Sp:{train_metrics['spearman']:.4f}, NDCG@20:{train_metrics['ndcg@20']:.4f} | Valid Sp:{val_metrics['spearman']:.4f}, NDCG@20:{val_metrics['ndcg@20']:.4f}")
            for key in history_keys:
                metrics_history['train'][key].append(train_metrics[key])
                metrics_history['val'][key].append(val_metrics[key])
            global_epoch_counter += 1
    else:
        if not args.use_pretrained:
            print("\n--- Starting training from scratch (single stage) ---")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.freeze_encoder:
            print("\n--- Starting single-stage frozen fine-tuning ---")
            optimizer = torch.optim.AdamW(model.mlp_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            print("\n--- Starting single-stage full fine-tuning ---")
            model.unfreeze_encoder_params()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        for epoch in tqdm(range(args.epochs), desc="Single-stage Training Progress", unit="epoch"):
            run_one_finetune_epoch(model, train_loader, optimizer, loss_func, device, is_train=True)
            train_metrics, _, _ = run_one_finetune_epoch(model, train_loader, None, loss_func, device, is_train=False)
            val_metrics, _, _ = run_one_finetune_epoch(model, valid_loader, None, loss_func, device, is_train=False)
            scheduler.step()
            print(
                f"Epoch {epoch + 1:03d} | Train Sp:{train_metrics['spearman']:.4f}, NDCG@20:{train_metrics['ndcg@20']:.4f} | Valid Sp:{val_metrics['spearman']:.4f}, NDCG@20:{val_metrics['ndcg@20']:.4f}")
            for key in history_keys:
                metrics_history['train'][key].append(train_metrics[key])
                metrics_history['val'][key].append(val_metrics[key])
            if val_metrics['ndcg@20'] > best_val_metric:
                best_val_metric = val_metrics['ndcg@20']
                torch.save(model.state_dict(), os.path.join(paths['run_path'], "best_model_state.pt"))
                print(f"  -> New best model saved (Validation NDCG@20: {best_val_metric:.4f})")

    last_model_path = os.path.join(paths['run_path'], "last_model_state.pt")
    torch.save(model.state_dict(), last_model_path)
    print(f"\n--- Training finished, model from the last epoch saved to: {last_model_path} ---")

    print("\n--- Loading the model from the [last epoch] for final evaluation... ---")
    model.load_state_dict(torch.load(os.path.join(paths['run_path'], "last_model_state.pt")))

    final_train_metrics, _, _ = run_one_finetune_epoch(model, train_loader, None, loss_func, device, is_train=False)
    final_val_metrics, val_preds, val_targets = run_one_finetune_epoch(model, valid_loader, None, loss_func, device,
                                                                       is_train=False)

    print(f"\nFinal validation set metrics (based on the model from the [last epoch]):")
    for key, value in final_val_metrics.items():
        print(f"  - {key.capitalize()}: {value:.4f}")

    print("\n--- Generating plots and saving results ---")
    plot_loss_curves(metrics_history['train']['loss'], metrics_history['val']['loss'], 'Loss Curve', paths['plot_path'],
                     "loss_curve.png")
    plot_spearman_curves(metrics_history['train']['spearman'], metrics_history['val']['spearman'],
                         'Spearman Correlation Curve', paths['plot_path'], "spearman_curve.png")
    plot_ndcg_curves(metrics_history['train']['ndcg@20'], metrics_history['val']['ndcg@20'], 20, paths['plot_path'],
                     "ndcg20_curve.png")

    scatter_plot_basic(val_preds, val_targets, "Validation Set: Predicted vs. True Values (last epoch model)", paths['plot_path'],
                       "scatter_val_last_model.png")

    pretrain_results = {}
    if args.use_pretrained:
        pretrain_results = load_pretrain_summary(args.pretrained_path)

    excel_filepath = os.path.join(paths['excel_path'], "summary.xlsx")
    memo_str = group_name
    for split, metrics in [('train', final_train_metrics), ('val', final_val_metrics)]:
        lr_finetune_val = "N/A"
        if args.two_stage_finetune and args.use_pretrained:
            lr_finetune_val = f"h:{args.lr_finetune_head}, e:{args.lr_finetune_encoder}"

        run_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'experiment_group': memo_str, 'dataset_split': split,
            'seed': args.seed,
            'pretrain_samples': args.sample_lines if args.use_pretrained else 'N/A',
            'pretrain_label_scaling': args.pretrain_label_scaling if args.use_pretrained else 'N/A',
            'lr_pretrain': args.lr_pretrain if args.use_pretrained else 'N/A',
            'bs_pretrain': args.pretrain_bs if args.use_pretrained else 'N/A',
            'pretrain_loss': args.loss_func_pretrain if args.use_pretrained else 'N/A',
            'pt_val_spearman': pretrain_results.get('spearman', 'N/A'),
            'pt_val_loss': pretrain_results.get('loss', 'N/A'),
            'pt_val_mae': pretrain_results.get('mae', 'N/A'),
            'pt_val_r2': pretrain_results.get('r2', 'N/A'),
            'downstream_train_samples': args.num_train_samples, 'downstream_val_samples': args.num_val_samples,
            'lr_stage1_or_scratch': args.lr, 'lr_stage2_finetune': lr_finetune_val,
            'batch_size': args.batch_size,
            'epochs_scratch': args.epochs if not args.two_stage_finetune else 'N/A',
            'epochs_stage1': args.epochs_head_only if args.two_stage_finetune else 'N/A',
            'epochs_stage2': args.epochs_full_finetune if args.two_stage_finetune else 'N/A',
            'downstream_loss': args.loss_func, 'use_pretrained': args.use_pretrained,
            'freeze_encoder': args.freeze_encoder,
            'two_stage_finetune': args.two_stage_finetune, 'model_name': args.model_name,
            'num_layers': args.num_layers, 'num_heads': args.num_heads, 'hidden_dim': args.hidden_dim,
            'encoder_dropout': args.dropout, 'mlp_dropout': args.dropout_mlp, 'weight_decay': args.weight_decay,
            **metrics
        }
        column_order = [
            'timestamp', 'experiment_group', 'dataset_split', 'seed',
            'use_pretrained', 'freeze_encoder', 'two_stage_finetune',
            'pretrain_samples', 'pretrain_label_scaling', 'lr_pretrain', 'bs_pretrain', 'pretrain_loss',
            'pt_val_spearman', 'pt_val_loss', 'pt_val_mae', 'pt_val_r2',
            'downstream_train_samples', 'downstream_val_samples',
            'lr_stage1_or_scratch', 'lr_stage2_finetune',
            'batch_size', 'epochs_scratch', 'epochs_stage1', 'epochs_stage2',
            'downstream_loss',
            'model_name', 'num_layers', 'num_heads', 'hidden_dim',
            'encoder_dropout', 'mlp_dropout', 'weight_decay',
            'loss', 'spearman', 'ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20',
            'ndcg@50', 'ndcg@100',
            'top1', 'top5', 'top10', 'top20', 'top50', 'top100'
        ]
        log_to_excel(filepath=excel_filepath, run_data=run_data, column_order=column_order)

    print(f"Results have been logged to: {excel_filepath}")
    print(f"\nBest validation set NDCG@20 during training: {best_val_metric:.4f}")
    print(f'Total fine-tuning time: {format_execution_time(time.time() - start_time)}')


# ==============================================================================
#                             Main Program Entry Point
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QuantumCircuitRanker - A unified training script for quantum circuit ranking models.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'finetune'],
                        help="Select execution mode: 'pretrain' or 'finetune'.")

    common_args = parser.add_argument_group('Common Model and Task Parameters')
    common_args.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    common_args.add_argument('--task_name', type=str, default='Heisenberg_8', choices=['Heisenberg_8', 'TFIM_8', 'TFCluster_8'],
                             help='Identifier for the downstream task (also required for pre-training).')
    common_args.add_argument("--model_name", type=str, default="DAGTransformer", help="Model name.")
    common_args.add_argument("--num_layers", type=int, default=3, help="Number of layers in the Transformer encoder.")
    common_args.add_argument("--num_heads", type=int, default=1, help="Number of heads in the multi-head attention mechanism.")
    common_args.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the encoder.")
    common_args.add_argument("--input_dim", type=int, default=13, help="Dimensionality of node features.")
    common_args.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of the Transformer (d_model).")

    pretrain_args = parser.add_argument_group('Pre-training Specific Parameters')
    pretrain_args.add_argument('--pretrain_label_scaling', type=str, default='standard-minmax',
                               choices=['none', 'standard-minmax', 'quantile'],
                               help="Normalization strategy for pre-training labels: 'none', 'standard-minmax', 'quantile'.")
    pretrain_args.add_argument("--sample_lines", type=int, default=10000, help="Number of circuit samples for pre-training.")
    pretrain_args.add_argument("--epochs_pretrain", type=int, default=20, help="Number of epochs for pre-training.")
    pretrain_args.add_argument("--pretrain_bs", type=int, default=64, help="Batch size for pre-training.")
    pretrain_args.add_argument("--lr_pretrain", type=float, default=1e-4, help="Learning rate for pre-training.")
    pretrain_args.add_argument("--val_split", type=float, default=0.2, help="Proportion of the pre-training data to use for validation.")
    pretrain_args.add_argument('--loss_func_pretrain', type=str, default='huber', choices=['mse', 'huber'],
                               help="Loss function to use for pre-training.")

    finetune_args = parser.add_argument_group('Fine-tuning Specific Parameters')
    finetune_args.add_argument('--use_pretrained', type=lambda x: x.lower() == 'true', default=True,
                               help="Whether to load pre-trained weights for fine-tuning.")
    finetune_args.add_argument('--two_stage_finetune', type=lambda x: x.lower() == 'true', default=True,
                               help="Whether to enable the two-stage fine-tuning strategy.")
    finetune_args.add_argument('--freeze_encoder', type=lambda x: x.lower() == 'true', default=True,
                               help="[Effective only in single-stage fine-tuning] Whether to freeze the encoder.")
    finetune_args.add_argument('--pretrained_path', type=str, default=None,
                               help='Path to the pre-trained encoder weights. If None, it will be constructed automatically.')
    finetune_args.add_argument('--loss_func', type=str, default='mse', choices=['mse', 'SoftNDCG'],
                               help="Loss function for the downstream task.")
    finetune_args.add_argument('--num_train_samples', type=int, default=200, help="Number of training samples for the downstream task.")
    finetune_args.add_argument('--num_val_samples', type=int, default=300, help="Number of validation samples for the downstream task.")
    finetune_args.add_argument("--batch_size", type=int, default=32, help="Batch size for the downstream task.")
    finetune_args.add_argument('--dropout_mlp', type=float, default=0.5,
                               help="Dropout rate for the downstream MLP head (recommended to lower for small samples).")
    finetune_args.add_argument('--lr', type=float, default=1e-4,
                               help="Learning rate (for training from scratch / single-stage fine-tuning / stage 1 of two-stage).")
    finetune_args.add_argument('--epochs', type=int, default=200, help="[Effective only in single-stage training] Total number of training epochs.")
    finetune_args.add_argument('--lr_finetune_head', type=float, default=1e-4, help="[Two-stage] Learning rate for the MLP head in stage 2.")
    finetune_args.add_argument('--lr_finetune_encoder', type=float, default=1e-6,
                               help="[Two-stage] Learning rate for the Encoder in stage 2 (should be very small).")
    finetune_args.add_argument('--epochs_head_only', type=int, default=50, help='[Two-stage] Number of epochs for stage 1 (training the head only).')
    finetune_args.add_argument('--epochs_full_finetune', type=int, default=50,
                               help='[Two-stage] Number of epochs for stage 2 (full fine-tuning).')
    finetune_args.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for the optimizer (L2 regularization).")

    ndcg_args = parser.add_argument_group('SoftNDCG Specific Parameters')
    ndcg_args.add_argument("--k", type=int, default=-1, help="The k value for SoftNDCG, -1 means using all samples.")
    ndcg_args.add_argument("--metric_function", type=str, default='L1', help="The internal distance metric for SoftNDCG.")
    ndcg_args.add_argument("--temperature", type=float, default=0.1, help="The temperature parameter for SoftNDCG.")

    args = parser.parse_args()

    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(f"Hidden dimension ({args.hidden_dim}) must be divisible by the number of attention heads ({args.num_heads}).")

    if args.mode == 'finetune' and args.use_pretrained and args.pretrained_path is None:
        base_path = "pretrain_results"
        hyperparam_str = f"pbs{args.pretrain_bs}_plr{args.lr_pretrain:.0e}_ploss-{args.loss_func_pretrain}_s{args.sample_lines}"
        exp_path = os.path.join(base_path, args.task_name, hyperparam_str, f"seed{args.seed}")
        args.pretrained_path = os.path.join(exp_path, 'models', "last_encoder.pt")
        print(f"Automatically constructing pre-trained model path for task '{args.task_name}': {args.pretrained_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfgs = TASK_CONFIGS.get(args.task_name, {})
    print(f"--- Task Configuration: {args.task_name}, Config: {cfgs} ---")

    if args.mode == 'pretrain':
        run_pretraining(args, cfgs)
    elif args.mode == 'finetune':
        run_finetuning(args, cfgs)
    else:
        print(f"Error: Unknown mode '{args.mode}'. Please choose from 'pretrain', 'finetune'.")

    print("\n--- Script execution finished ---")