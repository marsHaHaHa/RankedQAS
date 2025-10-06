import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorcircuit as tc

# Assuming these are custom modules for your project
from energy_calculator_TFCluster import calculate_true_energies_TFCluster
from energy_calculator import calculate_true_energies
from utils.common import ensure_dir, load_pkl, save_pkl, format_execution_time
from utils import graph_utils as utils_graph
from config import TASK_CONFIGS
from model.DAGTransformer import DAGTransformer
from model.MLP import MLP, SqueezeOutput
from model.losses.SoftNDCGLoss import SoftNDCGLoss
from quantum_gates import Gate


def pad_adj_ops(adj_list, ops_list, max_nodes):
    """Pad the adjacency matrix and operation features to the maximum number of nodes in the dataset."""
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


class DownstreamModel(torch.nn.Module):
    """A complete model for downstream tasks, including logic for freezing/unfreezing the encoder."""

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
            graph_embedding = self.encoder(data)

        prediction = self.mlp_head(graph_embedding)
        return self.squeeze(self.sigmoid(prediction))

    def freeze_encoder_params(self):
        """Freeze all parameters of the encoder."""
        self.is_encoder_frozen = True
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder_params(self):
        """Unfreeze all parameters of the encoder."""
        self.is_encoder_frozen = False
        for param in self.encoder.parameters():
            param.requires_grad = True


def build_QAS_model(args):
    """Build a downstream task model for fine-tuning."""
    encoder = DAGTransformer(in_dim=args.input_dim, d_model=args.hidden_dim, num_heads=args.num_heads,
                             num_layers=args.num_layers, dropout=args.dropout)
    mlp_head = MLP(input_dim=args.hidden_dim, hidden_dims=[64, 32], output_dim=1,
                   activation='relu', use_bn=True, dropout=args.dropout_mlp)
    should_freeze = args.use_pretrained and (args.two_stage_finetune or args.freeze_encoder)
    model = DownstreamModel(encoder, mlp_head, freeze_encoder=should_freeze)
    return model


def _process_single_graph(args):
    """
    A helper function for processing individual graph data, designed for parallelization.
    Accepts a tuple as a parameter, which contains all the information needed to process a single graph.
    """
    adj_orig, ops_padded, adj_padded, max_nodes = args

    num_real_nodes = adj_orig.shape[0]
    edge_index = utils_graph.adj2edge_index(adj_orig)
    temp_data = Data(edge_index=edge_index, num_nodes=num_real_nodes)
    real_depths = utils_graph.get_node_depths(temp_data)
    reachability_edge_index = utils_graph.get_reachability_edge_index(temp_data)

    padded_depths = torch.zeros(max_nodes, dtype=torch.long)
    padded_depths[:num_real_nodes] = real_depths
    mask = torch.zeros(max_nodes, dtype=torch.bool)
    mask[:num_real_nodes] = True

    return Data(x=torch.Tensor(ops_padded),
                adj=torch.Tensor(adj_padded),
                edge_index=edge_index,
                depths=padded_depths,
                mask=mask,
                reachability_edge_index=reachability_edge_index)


def load_finetune_data(args):
    """Load and prepare data for fine-tuning."""

    def get_or_create_processed_data(seed, task_name, num_workers=None):
        """
        Get or create a list of processed PyG data.
        If the cache file exists, load it directly; otherwise, process the raw data in parallel and create the cache.

        Args:
            seed (int): Random seed.
            task_name (str): Task name.
            num_workers (int, optional): Number of CPU cores used for parallel processing.
                                         If None, defaults to the number of CPU cores in the system.
        """
        data_path = f'data/raw/grid_16q_{task_name}/search_pool/run_{seed}'
        cache_path = os.path.join(data_path, 'processed_dag_data_QAS.pkl')
        if os.path.exists(cache_path):
            print(f"Loading processed data from cache: {cache_path}")
            return load_pkl(cache_path)

        print(f"Creating cached data for seed {seed} of task '{task_name}'...")
        matrix_all = load_pkl(f"{data_path}/adj_feat_matrix.pkl")
        matrix = matrix_all
        adj_list_orig, ops_list_orig = [c[0] for c in matrix], [c[1] for c in matrix]
        max_nodes = max(adj.shape[0] for adj in adj_list_orig) if adj_list_orig else 0
        adj_padded_list, ops_padded_list = pad_adj_ops(adj_list_orig, ops_list_orig, max_nodes)

        # --- Start of parallel processing section ---
        # 1. Determine the number of worker processes
        if num_workers is None:
            num_workers = multiprocessing.cpu_count() - 10  # Leave some cores for the system
        print(f"Using {num_workers} worker processes for parallel data processing...")

        # 2. Prepare the arguments for each worker process
        #    Use zip to create an iterator to save memory
        tasks = zip(adj_list_orig, ops_padded_list, adj_padded_list, [max_nodes] * len(matrix))

        # 3. Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # executor.map will pass each element in tasks to _process_single_graph in order
            # We wrap it with tqdm to display a progress bar
            results_iterator = executor.map(_process_single_graph, tasks)
            pyg_list = list(tqdm(results_iterator, total=len(matrix), desc=f"Parallel processing for seed {seed}"))

        # --- End of parallel processing section ---

        print(f"Data processing completed for seed {seed}. Saving to cache: {cache_path}")
        save_pkl(pyg_list, cache_path)
        return pyg_list

    num_parallel_workers = max(1, multiprocessing.cpu_count() - 5)

    test_data_full = get_or_create_processed_data(args.seed, args.task_name, num_workers=num_parallel_workers)
    test_data = test_data_full
    print(f"Using {len(test_data)} test samples.")

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

    circuits_path = f'data/raw/grid_16q_{args.task_name}/search_pool/run_{args.seed}/samples.pkl'
    QAS_circuits = load_pkl(circuits_path)

    return test_loader, QAS_circuits


def calculate_top_k_performance(y_pred, y_true, k_values):
    order = np.argsort(y_pred)[::-1]
    top_k_results = {}
    for k in k_values:
        if k > len(y_pred): continue
        top_k_indices = order[:k]
        best_energy_in_top_k = np.min(y_true[top_k_indices])
        top_k_results[f'Top-{k}_Energy'] = best_energy_in_top_k
    return top_k_results


def convert_gate_list(raw_gate_list):
    gate_objects = []
    for item in raw_gate_list:
        name = item[0]
        name_map = {
            'ry': 'Ry', 'rz': 'Rz', 'rx': 'Rx', 'cz': 'CZ', 'cx': 'CNOT'
        }
        name = name_map.get(name, name)
        qubit_indices = item[1:]
        qubits = len(qubit_indices)
        para_gate = (name in ['Rz', 'Ry', 'Rx'])
        gate = Gate(qubits=qubits, name=name, para_gate=para_gate)
        gate.act_on = qubit_indices
        gate_objects.append(gate)
    return gate_objects


def QAS(args, cfgs):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    base_path = f"finetune_results"
    mode_str_en = "2stage"
    lr_str = f"lr_h{args.lr_finetune_head:.0e}_e{args.lr_finetune_encoder:.0e}"
    finetune_params_str = f"ft_T{args.num_train_samples}_B{args.batch_size}_{lr_str}_L-{args.loss_func}"
    pretrain_params_str = f"pt_pbs{args.pretrain_bs}_plr{args.lr_pretrain:.0e}_ploss-{args.loss_func_pretrain}_s{args.sample_lines}"
    group_name = f"{mode_str_en}_{finetune_params_str}_{pretrain_params_str}"
    run_name = f"seed{args.seed}"
    model_path = os.path.join(base_path, args.task_name, group_name, run_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model path '{model_path}' does not exist. Please check if pre-training or fine-tuning has been completed.")
    print(f"Model path: {model_path}")

    # Model instantiation
    model = build_QAS_model(args).to(device)
    model.load_state_dict(torch.load(model_path + '/last_model_state.pt'))
    print(f"TASK: {args.task_name}, SEED: {args.seed}, Model loaded successfully!")

    # Load QAS dataset
    test_loader, QAS_circuits = load_finetune_data(args)

    all_preds = []
    model.eval()
    print(f"Model parameters fixed, starting QAS prediction...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="QAS prediction progress", total=len(test_loader)):
            batch = batch.to(device)
            pred_norm = model(batch)
            all_preds.extend(pred_norm.cpu().detach().numpy().flatten())

    predictions = np.array(all_preds)
    print(f"Prediction length: {len(predictions)}")

    # Extract indices from the predicted values in descending order.
    ranked_indices = np.argsort(predictions)[::-1]

    # Task 1: Select the top_k_save circuits from the sorted results for evaluation.
    print(f"\n--- Task 1: Selecting the top-{args.top_k_save} circuits for evaluation ---")
    top_k_to_save = args.top_k_save
    selected_indices = ranked_indices[:top_k_to_save]
    sorted_circuits_to_evaluate = [QAS_circuits[i] for i in selected_indices]
    sorted_circuits_to_evaluate = [convert_gate_list(circuit[0]) for circuit in sorted_circuits_to_evaluate]

    # Task 1: Evaluate the selected circuits using the external function.
    print(f"\n--- Task 1: Evaluating the top-{args.top_k_save} circuits ---")
    result_dir = f"experiment_results/QAS_results/{args.task_name}/seed-{args.seed}/"
    ensure_dir(result_dir)

    # Call an external function to calculate energy.
    hamiltonian = {}
    if args.task_name == 'Heisenberg_8':
        hamiltonian = {'pbc': True, 'hzz': 1, 'hxx': 1, 'hyy': 1, 'hx': 0, 'hy': 0, 'hz': 1, 'sparse': False}
    elif args.task_name == 'TFIM_8':
        hamiltonian = {'pbc': True, 'hzz': 1, 'hxx': 0, 'hyy': 0, 'hx': 1, 'hy': 0, 'hz': 0, 'sparse': False}
    elif args.task_name == 'TFCluster_8':
        def convert_pauli_strings_to_numerical(pauli_strings):
            """Helper function: Convert Pauli strings to TensorCircuit's numerical format."""
            pauli_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
            return [[pauli_map[char.upper()] for char in p_str] for p_str in pauli_strings]

        n_qubit = args.num_qubits
        hamiltonian_terms = []
        # Add boundary term 1: -X₀ Z₁
        term_list_b1 = ['I'] * n_qubit;
        term_list_b1[0] = 'X';
        term_list_b1[1] = 'Z'
        hamiltonian_terms.append(
            {"string": "".join(term_list_b1), "coeff": -1.0, "type": "boundary term", "indices": (0, 1)})
        # Build the main part: - sum_{j=1}^{n-2} Z_{j-1} X_j Z_{j+1}
        for j in range(1, n_qubit - 1):
            term_list = ['I'] * n_qubit;
            term_list[j - 1] = 'Z';
            term_list[j] = 'X';
            term_list[j + 1] = 'Z'
            hamiltonian_terms.append(
                {"string": "".join(term_list), "coeff": -1.0, "type": "main term", "indices": (j - 1, j, j + 1)})
        # Add boundary term 2: -Z_{n-2} X_{n-1}
        term_list_b2 = ['I'] * n_qubit;
        term_list_b2[n_qubit - 2] = 'Z';
        term_list_b2[n_qubit - 1] = 'X'
        hamiltonian_terms.append({"string": "".join(term_list_b2), "coeff": -1.0, "type": "boundary term",
                                  "indices": (n_qubit - 2, n_qubit - 1)})

        # --- Printing section: Sort by position and output ---
        sorted_terms_for_printing = sorted(hamiltonian_terms, key=lambda term: term["indices"][0])
        print("Pauli terms defined using strings (sorted by physical positions):")
        for term in sorted_terms_for_printing:
            print(f"- {term['string']} ({term['type']}, acting on {', '.join(map(str, term['indices']))})")

        # --- Calculation section ---
        pauli_strings_str = [term["string"] for term in hamiltonian_terms]
        coeffs = [term["coeff"] for term in hamiltonian_terms]
        pauli_terms_numerical = convert_pauli_strings_to_numerical(pauli_strings_str)
        pauli_terms_tf = tf.constant(pauli_terms_numerical, dtype=tf.int32)
        coeffs_tf = tf.constant(coeffs, dtype=tf.complex128)
        hamiltonian_cluster_model = tc.quantum.PauliStringSum2Dense(pauli_terms_tf, coeffs_tf)
        print("\nSuccessfully constructed the Hamiltonian of the transverse-field cluster model.")
        print("Hamiltonian matrix (dense form) shape:", hamiltonian_cluster_model.shape)

    if args.task_name in ['Heisenberg_8', 'TFIM_8']:
        calculate_true_energies(
            seed=args.seed, qubit=args.num_qubits, task_name=args.task_name,
            circuits=sorted_circuits_to_evaluate, experiment_dir=result_dir,
            Hamiltonian=hamiltonian, noise=True,
        )
    elif args.task_name == 'TFCluster_8':
        calculate_true_energies_TFCluster(
            seed=args.seed, qubit=args.num_qubits, task_name=args.task_name,
            circuits=sorted_circuits_to_evaluate, experiment_dir=result_dir,
            hamiltonian_matrix=hamiltonian_cluster_model, noise=True,
        )

    evaluated_true_energies = np.array(load_pkl(os.path.join(result_dir, 'energy.pkl')))
    print(f"\n--- Saving detailed information of the top {args.top_k_save} circuits ---")

    results_to_save = []
    for i in range(len(selected_indices)):
        original_index = selected_indices[i]
        result_item = {
            'rank': i + 1,
            'original_index': original_index,
            'predicted_score': predictions[original_index],
            'true_energy': evaluated_true_energies[i],
            'circuit_structure': sorted_circuits_to_evaluate[i],
            'gate_count': len(sorted_circuits_to_evaluate[i])
        }
        results_to_save.append(result_item)

    save_path_pkl = os.path.join(result_dir, f'selected_top_{top_k_to_save}_results.pkl')
    save_pkl(results_to_save, save_path_pkl)
    print(f"Detailed results have been saved to (PKL): {save_path_pkl}")

    # Task 3: Calculate the test set metrics
    y_pred_for_eval = predictions[selected_indices]
    y_true_for_eval = evaluated_true_energies

    # Top-k Energy
    top_k_values = [1, 5, 10, 15, 20, 50, 100, 200, 300]
    top_k_results = calculate_top_k_performance(y_pred=y_pred_for_eval, y_true=y_true_for_eval, k_values=top_k_values)
    print("Top-k energy metric (best energy among the k circuits selected by the model):")
    for key, value in top_k_results.items():
        print(f"  - {key}: {value:.6f}")

    qas_found_energy = evaluated_true_energies[0]
    ground_truth_in_evaluated = np.min(evaluated_true_energies)
    print(f"Optimal energy found by QAS (Rank 1): {qas_found_energy:.6f}")
    print(f"Ground state energy in the evaluation subset: {ground_truth_in_evaluated:.6f}")

    # Task 4: Save all metrics to Excel
    excel_path = f"experiment_results/QAS_results/{args.task_name}/{args.task_name}_QAS_results.xlsx"
    excel_row_data = {
        'Task': args.task_name, 'Qubits': args.num_qubits, 'Seed': args.seed,
        'QAS_Found_Energy': qas_found_energy,
        'Ground_Truth_Energy_in_Elites': ground_truth_in_evaluated,
        **top_k_results
    }
    if os.path.exists(excel_path):
        df_excel = pd.read_excel(excel_path)
        df_excel = pd.concat([df_excel, pd.DataFrame([excel_row_data])], ignore_index=True)
    else:
        df_excel = pd.DataFrame([excel_row_data])

    df_excel.to_excel(excel_path, index=False)
    print(f"All metrics have been summarized and saved to: {excel_path}")

    print(f'\n---------Final Results Summary--------')
    print(f"Task: {args.task_name}, Qubits: {args.num_qubits}, Seed: {args.seed}")
    print(f"Best energy found by QAS (Rank 1): {qas_found_energy:.6f}")
    print(f"Ground state energy in the evaluated subset: {ground_truth_in_evaluated:.6f}")
    print(f"Top-k Energy Metrics:")
    for key, value in top_k_results.items():
        print(f"  - {key}: {value:.6f}")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print('------------------------------------')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="QuantumCircuitRanker - A unified script for quantum circuit ranking models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=str, default='QAS', choices=['pretrain', 'finetune', 'QAS'],
                        help="Select execution mode: 'pretrain', 'finetune', or 'QAS' (Quantum Architecture Search).")

    common_args = parser.add_argument_group('Common Model and Task Parameters')
    parser.add_argument('--top_k_save', type=int, default=10, help='Number of top circuits to save and evaluate.')
    parser.add_argument('--noise', type=bool, default=True, help='Whether to simulate noise in energy calculation.')
    parser.add_argument("--num_qubits", type=int, default=8, help="Number of qubits.")
    common_args.add_argument("--seed", type=int, default=2, help="Random seed for reproducibility.")
    common_args.add_argument('--task_name', type=str, default='TFCluster_8',
                             choices=['Heisenberg_8', 'TFIM_8', 'TFCluster_8'],
                             help='Identifier for the downstream task (also required for pre-training).')
    common_args.add_argument("--model_name", type=str, default="DAGTransformer", help="Model name.")
    common_args.add_argument("--num_layers", type=int, default=3, help="Number of layers in the Transformer encoder.")
    common_args.add_argument("--num_heads", type=int, default=1,
                             help="Number of heads in the multi-head attention mechanism.")
    common_args.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the encoder.")
    common_args.add_argument("--input_dim", type=int, default=13, help="Dimensionality of node features.")
    common_args.add_argument("--hidden_dim", type=int, default=64,
                             help="Hidden dimension of the Transformer (d_model).")

    pretrain_args = parser.add_argument_group('Pre-training Specific Parameters')
    pretrain_args.add_argument("--sample_lines", type=int, default=10000,
                               help="Number of circuit samples for pre-training.")
    pretrain_args.add_argument("--epochs_pretrain", type=int, default=3, help="Number of epochs for pre-training.")
    pretrain_args.add_argument("--pretrain_bs", type=int, default=32, help="Batch size for pre-training.")
    pretrain_args.add_argument("--lr_pretrain", type=float, default=1e-4, help="Learning rate for pre-training.")
    pretrain_args.add_argument("--val_split", type=float, default=0.2,
                               help="Proportion of the pre-training data to use for validation.")
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
    finetune_args.add_argument('--loss_func', type=str, default='SoftNDCG', choices=['mse', 'SoftNDCG'],
                               help="Loss function for the downstream task.")
    finetune_args.add_argument('--num_train_samples', type=int, default=200,
                               help="Number of training samples for the downstream task.")
    finetune_args.add_argument('--num_val_samples', type=int, default=300,
                               help="Number of validation samples for the downstream task.")
    finetune_args.add_argument("--batch_size", type=int, default=32, help="Batch size for the downstream task.")
    finetune_args.add_argument('--dropout_mlp', type=float, default=0.5,
                               help="Dropout rate for the downstream MLP head (recommended to lower for small samples).")
    finetune_args.add_argument('--lr', type=float, default=1e-4,
                               help="Learning rate (for training from scratch / single-stage fine-tuning / stage 1 of two-stage).")
    finetune_args.add_argument('--epochs', type=int, default=200,
                               help="[Effective only in single-stage training] Total number of training epochs.")
    finetune_args.add_argument('--lr_finetune_head', type=float, default=1e-3,
                               help="[Two-stage] Learning rate for the MLP head in stage 2.")
    finetune_args.add_argument('--lr_finetune_encoder', type=float, default=1e-4,
                               help="[Two-stage] Learning rate for the Encoder in stage 2 (should be smaller than head's LR).")
    finetune_args.add_argument('--epochs_head_only', type=int, default=50,
                               help='[Two-stage] Number of epochs for stage 1 (training the head only).')
    finetune_args.add_argument('--epochs_full_finetune', type=int, default=50,
                               help='[Two-stage] Number of epochs for stage 2 (full fine-tuning).')
    finetune_args.add_argument('--weight_decay', type=float, default=1e-5,
                               help="Weight decay for the optimizer (L2 regularization).")

    ndcg_args = parser.add_argument_group('SoftNDCG Specific Parameters')
    ndcg_args.add_argument("--k", type=int, default=-1, help="The k value for SoftNDCG, -1 means using all samples.")
    ndcg_args.add_argument("--metric_function", type=str, default='L1',
                           help="The internal distance metric for SoftNDCG.")
    ndcg_args.add_argument("--temperature", type=float, default=0.1, help="The temperature parameter for SoftNDCG.")

    args = parser.parse_args()

    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(
            f"Hidden dimension ({args.hidden_dim}) must be divisible by the number of attention heads ({args.num_heads}).")

    # [MODIFIED] Automatically construct the pre-trained model path to match the new hierarchical structure with hyperparameters.
    if args.mode == 'finetune' and args.use_pretrained and args.pretrained_path is None:
        base_path = "pretrain_results"
        hyperparam_str = f"pbs{args.pretrain_bs}_plr{args.lr_pretrain:.0e}_ploss-{args.loss_func_pretrain}_s{args.sample_lines}"
        exp_path = os.path.join(base_path, args.task_name, hyperparam_str, f"seed{args.seed}")
        # Default to loading the model from the last epoch
        args.pretrained_path = os.path.join(exp_path, 'models', "last_encoder.pt")
        print(f"Automatically constructing pre-trained model path for task '{args.task_name}': {args.pretrained_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfgs = TASK_CONFIGS.get(args.task_name, {})

    QAS(args, cfgs)