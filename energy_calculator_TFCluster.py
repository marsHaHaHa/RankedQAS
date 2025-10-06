import argparse
import os
import random
from quantum_gates import  Gate
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import re
from utils.common import load_pkl
import utils.common as utils

tc.set_dtype("complex128")
tc.set_backend("tensorflow")

def convert_to_list_representation(circuit_of_gates):
    raw_gate_list = []
    for gate in circuit_of_gates:
        name_map = {
            'Ry': 'ry', 'Rz': 'rz', 'Rx': 'rx', 'CZ': 'cz', 'CNOT': 'cx'
        }
        name = name_map.get(gate.name, gate.name.lower())
        qubit_indices = gate.act_on
        if qubit_indices is None or not isinstance(qubit_indices, list):
            print(f"Warning: The act_on attribute of Gate '{gate.name}' is invalid and has been skipped.")
            continue
        item = [name] + qubit_indices
        raw_gate_list.append(item)
    return raw_gate_list


def depth_count(cir, qubit):

    def convert_gate_list(raw_gate_list):
        gate_objects = []
        for item in raw_gate_list:
            name_map = {
                'ry': 'Ry', 'rz': 'Rz', 'rx': 'Rx', 'cz': 'CZ', 'cx': 'CNOT'
            }
            name = name_map.get(item[0], item[0])
            qubit_indices = item[1:]
            qubits = len(qubit_indices)
            para_gate = (name in ['Rz', 'Ry', 'Rx'])
            gate = Gate(name=name, qubits=qubits, para_gate=para_gate)
            gate.act_on = qubit_indices
            gate_objects.append(gate)
        return gate_objects

    cir = convert_gate_list(cir)

    res = [0] * qubit
    for gate in cir:
        if gate.qubits > 1:
            depth_q = [res[q] for q in gate.act_on]
            max_depth = max(depth_q) + 1
            for q in gate.act_on:
                res[q] = max_depth
        else:
            if gate.act_on:
                res[gate.act_on[0]] += 1
    return np.max(res) if res else 0


class VqeTrainerNew:
    def __init__(self, n_cir_parallel, n_runs, max_iteration, n_qubit, hamiltonian_matrix, noise_param=None):
        self.K = tc.set_backend("tensorflow")
        self.n_qubit = n_qubit
        self.max_iteration = max_iteration
        self.n_cir_parallel = n_cir_parallel
        self.n_runs = n_runs

        self.h = hamiltonian_matrix

        if noise_param is None:
            self.noise = False
        else:
            self.noise = True
            self.two_qubit_channel_depolarizing_p = noise_param['two_qubit_channel_depolarizing_p']
            self.single_qubit_channel_depolarizing_p = noise_param['single_qubit_channel_depolarizing_p']
            self.bit_flip_p = noise_param['bit_flip_p']
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(
                self.two_qubit_channel_depolarizing_p / 15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(
                self.single_qubit_channel_depolarizing_p / 3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

    def compute_energy(self, param, structure):
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)
            c = tc.DMCircuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
                if gate[0] == "cx":
                    c.cx(gate[1], gate[2])
                    c.general_kraus(self.two_qubit_dep_channel, gate[1], gate[2])
                elif gate[0] == "cz":
                    c.cz(gate[1], gate[2])
                    c.general_kraus(self.two_qubit_dep_channel, gate[1], gate[2])
                elif gate[0] in ["ry", "rz", "rx"]:
                    getattr(c, gate[0])(gate[1], theta=param[param_index])
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    param_index += 1
                else:
                    raise ValueError(f"invalid gate: {gate[0]}")
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)
            st = c.state()
            e = self.K.real(tf.linalg.trace(st @ self.h))
        else:
            c = tc.Circuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
                if gate[0] in ["cx", "cz"]:
                    getattr(c, gate[0])(gate[1], gate[2])
                elif gate[0] in ["ry", "rz", "rx"]:
                    getattr(c, gate[0])(gate[1], theta=param[param_index])
                    param_index += 1
                else:
                    raise ValueError(f"invalid gate: {gate[0]}")
            e = tc.templates.measurements.operator_expectation(c, self.h)
        return e

    def get_param_num(self, cir):
        return sum(1 for gate in cir if gate[0] in ['rx', 'ry', 'rz'])

    def train_circuit(self, work_item):
        single_circuit, seed, circuit_index = work_item
        np.random.seed(seed)
        tf.random.set_seed(seed)

        param_num = self.get_param_num(single_circuit)
        trainer = tc.backend.jit(tc.backend.value_and_grad(self.compute_energy, argnums=0))
        L = depth_count(single_circuit, self.n_qubit)
        par = np.random.normal(loc=0, scale=1 / (8 * (L + 2)), size=param_num)
        param = tf.Variable(initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr)))

        e_last = 1000
        energy_epoch = []
        opt = tf.keras.optimizers.Adam(0.05)

        if param_num > 0:
            for i in range(self.max_iteration):
                e, grad = trainer(param, single_circuit)
                energy_epoch.append(e.numpy())
                opt.apply_gradients([(grad, param)])
                if i > 0 and i % 100 == 0:
                    distance = abs(e_last - e.numpy())
                    if distance < 0.0001:
                        break
                    e_last = e.numpy()
        else:
            e, _ = trainer(param, single_circuit)
            energy_epoch = [e.numpy()] * 2

        return e.numpy(), param.numpy(), energy_epoch, circuit_index

    def train_and_save_batch(self, circuit_batch, batch_index, batch_results_dir):
        print(f"--- Starting efficient parallel training batch {batch_index} (total of {len(circuit_batch)} circuits) ---")
        batch_start_time = time.time()

        work_queue = []
        for i, circuit in enumerate(circuit_batch):
            for run_seed in range(self.n_runs):
                work_queue.append((circuit, run_seed, i))

        print(f"--- Created {len(work_queue)} independent training tasks for batch {batch_index} ---")

        with Pool(processes=self.n_cir_parallel) as pool:
            all_results = pool.map(self.train_circuit, work_queue)

        grouped_results = [[] for _ in range(len(circuit_batch))]
        for energy, param, energy_epoch, circuit_idx in all_results:
            grouped_results[circuit_idx].append((energy, param, energy_epoch))

        print(f"--- All parallel training tasks for batch {batch_index} have completed ---")

        energy_batch, param_batch, energy_epoch_batch = [], [], []
        for i, single_circuit_results in enumerate(grouped_results):
            if not single_circuit_results:
                print(f"--- Warning: Batch {batch_index} circuit {i} did not receive any training results, possibly due to an error. ---")
                continue

            energies, params, energy_epochs = zip(*single_circuit_results)
            best_run_index = np.argmin(energies)

            energy_batch.append(energies[best_run_index])
            param_batch.append(params[best_run_index])
            energy_epoch_batch.append(energy_epochs[best_run_index])

        batch_result_data = {
            'energy': energy_batch,
            'param': param_batch,
            'energy_epoch': energy_epoch_batch,
        }

        save_path = os.path.join(batch_results_dir, f'results_batch_{batch_index}.pkl')
        utils.save_pkl(batch_result_data, save_path)

        batch_duration = time.time() - batch_start_time
        print(f"--- Batch {batch_index} training completed. Time taken: {batch_duration:.2f} seconds. Results saved to {save_path} ---")

    def run_batched_training(self, circuits, task_name, run_id, exp_dir, batch_size=100):

        print(f"\nFixing global randomness using seed {run_id}...")
        np.random.seed(run_id)
        tf.random.set_seed(run_id)
        random.seed(run_id)
        print(f"--- Global randomness fixed using seed {run_id}. Numpy, TensorFlow, and Random modules' seeds have been set. ---")

        exp_dir = exp_dir
        batch_results_dir = os.path.join(exp_dir, "batch_results")
        os.makedirs(batch_results_dir, exist_ok=True)

        print(f"--- Experiment directory: {exp_dir} ---")
        print(f"--- Batch results will be saved to: {batch_results_dir} ---")

        try:
            all_circuits = circuits
            all_circuits = [convert_to_list_representation(c) for c in all_circuits]
        except FileNotFoundError:
            print(f"--- Error: Could not find the circuit file {exp_dir}circuits.pkl. Please generate the circuits first. ---")
            return

        total_circuits = len(all_circuits)
        print(f"--- Total number of circuits to train: {total_circuits} ---")

        total_start_time = time.time()

        for i in range(0, total_circuits, batch_size):
            batch_index = i // batch_size
            circuit_batch = all_circuits[i:i + batch_size]

            batch_result_file = os.path.join(batch_results_dir, f'results_batch_{batch_index}.pkl')
            if os.path.exists(batch_result_file):
                print(f"--- Batch {batch_index} results file already exists. Skipping training. ---")
                continue

            self.train_and_save_batch(circuit_batch, batch_index, batch_results_dir)

        total_duration = time.time() - total_start_time
        print(
            f"--- All batches training completed. Total training time: {int(total_duration // 3600)} hours {int((total_duration % 3600) // 60)} minutes ---")
        utils.save_pkl(total_duration, os.path.join(exp_dir, 'duration_train.pkl'))

    def merge_results(self, task_name, run_id, exp_dir):
        exp_dir = exp_dir
        batch_results_dir = os.path.join(exp_dir, "batch_results")

        if not os.path.exists(batch_results_dir):
            print(f"Error: Batch results directory {batch_results_dir} not found.")
            return

        try:
            result_files = [f for f in os.listdir(batch_results_dir) if
                            f.startswith('results_batch_') and f.endswith('.pkl')]
            result_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group(1)))
        except (IOError, TypeError) as e:
            print(f"--- Error: Failed to read or sort batch result files in {batch_results_dir}. Error: {e} ---")
            return

        if not result_files:
            print(f"--- Warning: No batch result files found in {batch_results_dir} to merge. ---")
            return

        energy_final, param_final, energy_epoch_final = [], [], []

        for filename in result_files:
            filepath = os.path.join(batch_results_dir, filename)
            try:
                data = utils.load_pkl(filepath)
                energy_final.extend(data['energy'])
                param_final.extend(data['param'])
                energy_epoch_final.extend(data['energy_epoch'])
                print(f"  - Successfully loaded {filename}")
            except Exception as e:
                print(f"--- Warning: Failed to load or process file {filename} when merging results. Error: {e} ---")

        utils.save_pkl(energy_final, os.path.join(exp_dir, 'energy.pkl'))
        utils.save_pkl(param_final, os.path.join(exp_dir, 'param.pkl'))
        utils.save_pkl(energy_epoch_final, os.path.join(exp_dir, 'energy_epoch.pkl'))

        print(f"--- Successfully merged results from {len(result_files)} batch files. Total number of lines: {len(energy_final)} ---")
        print(f"--- Final merged files saved to: {exp_dir} ---")

def calculate_true_energies_TFCluster(
        seed,
        qubit,
        task_name,
        circuits,
        experiment_dir,
        hamiltonian_matrix=None,
        noise: bool=True,
        noise_param=None,
        device_name = 'grid_16q',
):
    run_id = seed
    device_name = device_name  # see config.py
    task_name = task_name
    qubit = qubit
    exp_dir = experiment_dir

    # 噪声信息
    two_qubit_depolarizing_p = 0.01
    single_qubit_depolarizing_p = 0.001
    bit_flip_p = 0.01
    if noise:
        noise_param = {'two_qubit_channel_depolarizing_p': two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': single_qubit_depolarizing_p,
                       'bit_flip_p': bit_flip_p}
    hamiltonian_matrix = hamiltonian_matrix
    if hamiltonian_matrix is None:
        raise ValueError(f"--- Error: Failed to load Hamiltonian matrix for molecule {task_name}. Please check the input Hamiltonian matrix. ---")

    print(f"run_id: {run_id}")
    print(f"qubit: {qubit}")
    print(f"device_name: {device_name}")
    print(f"task_name: {task_name}")
    print(f"noise: {noise}")
    print(f"noise_param: {noise_param}")
    print(f"hamiltonian_matrix: {hamiltonian_matrix.shape}")
    print(f"experiment_dir: {experiment_dir}")

    trainer = VqeTrainerNew(n_cir_parallel=10, n_runs=10, max_iteration=2000, n_qubit=qubit,
                            hamiltonian_matrix=hamiltonian_matrix,
                            noise_param=noise_param)

    print("\n========== Start the training step ==========")
    trainer.run_batched_training(circuits=circuits,task_name=task_name, run_id=run_id, exp_dir=exp_dir, batch_size=100)
    print("\n========== Start the merge step ==========")
    trainer.merge_results(task_name, run_id, exp_dir)





