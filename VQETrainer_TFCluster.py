import os
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import argparse
import utils.common as utils
import config

tc.set_dtype("complex128")
tc.set_backend("tensorflow")

class VqeTrainerNew:
    def __init__(self, n_cir_parallel, n_runs, max_iteration, n_qubit, hamiltonian, noise_param=None):
        self.K = tc.set_backend("tensorflow")
        self.n_qubit = n_qubit
        self.max_iteration = max_iteration
        self.n_cir_parallel = n_cir_parallel
        self.n_runs = n_runs
        self.h = hamiltonian  # Directly accept the pre-built Hamiltonian matrix

        self.give_up_rest = False
        self.solution = None

        """ Noise-related parameter, can be ignored if noise is False. """
        if noise_param is None:
            self.noise = False
        else:
            self.noise = True
        self.two_qubit_channel_depolarizing_p = None
        self.single_qubit_channel_depolarizing_p = None
        self.bit_flip_p = None
        if self.noise:
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
        """
        :param param: Circuit Parameters
        :param structure: Circuit structure
        :return: Energy expectation value
        """
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
                elif gate[0] == "ry":
                    c.ry(gate[1], theta=param[param_index])
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    param_index += 1
                elif gate[0] == "rz":
                    c.rz(gate[1], theta=param[param_index])
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    param_index += 1
                elif gate[0] == "rx":
                    c.rx(gate[1], theta=param[param_index])
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    param_index += 1
                else:
                    print("invalid gate!")
                    exit(0)
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            """Calculate energy"""
            st = c.state()
            x = tf.matmul(st, self.h)
            e = tf.linalg.trace(x)
            e = self.K.real(e)

        else:
            c = tc.Circuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
                if gate[0] == "cx":
                    c.cx(gate[1], gate[2])
                elif gate[0] == "cz":
                    c.cz(gate[1], gate[2])
                elif gate[0] == "ry":
                    c.ry(gate[1], theta=param[param_index])
                    param_index += 1
                elif gate[0] == "rz":
                    c.rz(gate[1], theta=param[param_index])
                    param_index += 1
                elif gate[0] == "rx":
                    c.rx(gate[1], theta=param[param_index])
                    param_index += 1
                else:
                    print("invalid gate!")
                    exit(0)
            e = tc.templates.measurements.operator_expectation(c, self.h)
        return e

    def get_param_num(self, cir):
        param_num = 0
        for i in range(len(cir)):
            if cir[i][0] == 'rx' or cir[i][0] == 'ry' or cir[i][0] == 'rz':
                param_num += 1
        return param_num

    def train_circuit(self, circuit_and_seed):
        single_circuit = circuit_and_seed[0]
        seed = circuit_and_seed[1]
        np.random.seed(seed)
        tf.random.set_seed(seed)

        param_num = self.get_param_num(single_circuit[0])
        trainer = tc.backend.jit(tc.backend.value_and_grad(self.compute_energy, argnums=0))
        L = single_circuit[1]
        par = np.random.normal(loc=0, scale=1 / (8 * (L + 2)), size=param_num)
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )
        e_last = 1000
        energy_epoch = []
        opt = tf.keras.optimizers.Adam(0.05)
        if param_num > 0:
            for i in range(self.max_iteration):
                e, grad = trainer(param, single_circuit[0])
                energy_epoch.append(e.numpy())
                opt.apply_gradients([(grad, param)])
                if i % 100 == 0:
                    distance = abs(e_last - e.numpy())
                    if distance < 0.0001:
                        break
                    else:
                        e_last = e.numpy()
        else:
            e, grad = trainer(param, single_circuit[0])
            energy_epoch = [e.numpy() for _ in range(self.max_iteration)]
        return e.numpy(), param.numpy(), energy_epoch

    def draw(self, loss_list, exp_dir, circuit_id, best_index):
        '''
        :param loss_list: list, the train loss of a single training
        :param exp_dir: the directory to save data, e.g., 'result/run_1/'
        :param circuit_id: which circuit
        :return:
        '''
        epochs = range(1, len(loss_list) + 1)
        plt.figure(figsize=(10, 6))
        # Plot the training loss
        plt.plot(epochs, loss_list, label='Training Loss', marker='o', markersize=1, color='blue')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        # Adjust layout and display the plot
        plt.tight_layout()
        save_path_img = f'{exp_dir}/circuit_train_curve/'
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        plt.savefig(save_path_img + f'circuit_{circuit_id}_{best_index}.png')
        plt.close()

    def batch_train_parallel(self, device_name, task_name, run_id):
        start_time = time.time()
        # load finetune circuits
        exp_dir = f'data/raw/{device_name}_{task_name}/training/run_{run_id}/'
        samples = utils.load_pkl(f'{exp_dir}samples.pkl')[:500]

        # start training
        work_queue = []
        for i in range(len(samples)):
            print(f'circuit id: {i}')
            # load finetune circuits
            work_queue.extend([[samples[i], j] for j in range(self.n_runs)])

        pool = Pool(processes=self.n_cir_parallel)
        result = pool.map(self.train_circuit, work_queue)
        pool.close()
        pool.join()

        energy, param, energy_epoch = [], [], []
        for part in result:
            energy.append(part[0])
            param.append(part[1])
            energy_epoch.append(part[2])

        energy_f, param_f, energy_epoch_f = [], [], []
        for i in range(len(samples)):
            index0 = i * self.n_runs
            index1 = index0 + self.n_runs
            best_index_local = np.argmin(energy[index0:index1])
            best_index_global = best_index_local + index0
            energy_f.append(energy[best_index_global])
            param_f.append(param[best_index_global])
            energy_epoch_f.append(energy_epoch[best_index_global])

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        utils.save_pkl(energy_f, f'{exp_dir}energy.pkl')
        utils.save_pkl(param_f, f'{exp_dir}param.pkl')
        utils.save_pkl(energy_epoch_f, f'{exp_dir}energy_epoch.pkl')
        end_time = time.time()
        duration = end_time - start_time
        print(f"run time: {int(duration // 3600)} hours {int((duration % 3600) // 60)} minutes")
        utils.save_pkl(duration, exp_dir + 'duration.pkl')


def convert_pauli_strings_to_numerical(pauli_strings):
    """
    Helper function: Converts a list of Pauli operators represented as strings
    to TensorCircuit's numerical format.
    For example: ['IX', 'ZI'] -> [[0, 1], [3, 0]]
    """
    pauli_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    numerical_terms = []
    for p_str in pauli_strings:
        numerical_terms.append([pauli_map[char.upper()] for char in p_str])
    return numerical_terms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=bool, default=True, help="Whether to consider noise in the simulation")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01, help="Noise level for the two-qubit depolarizing channel")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001, help="Noise level for the single-qubit depolarizing channel")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="Noise level for bit-flip errors")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    device_name = 'grid_16q'
    task_name = 'TFCluster_8'
    n_qubit = 8

    # -------- Construct Hamiltonian from strings with detailed metadata --------

    # We will create a list where each element contains all the information
    # needed for calculation and printing.
    hamiltonian_terms = []

    # Add boundary term 1: -X_0 Z_1
    term_list_b1 = ['I'] * n_qubit
    term_list_b1[0] = 'X'
    term_list_b1[1] = 'Z'
    hamiltonian_terms.append({
        "string": "".join(term_list_b1),
        "coeff": -1.0,
        "type": "Boundary Term",
        "indices": (0, 1)
    })

    # Build the bulk part: - sum_{j=1}^{n-2} Z_{j-1} X_j Z_{j+1}
    for j in range(1, n_qubit - 1):
        term_list = ['I'] * n_qubit
        term_list[j - 1] = 'Z'
        term_list[j] = 'X'
        term_list[j + 1] = 'Z'
        hamiltonian_terms.append({
            "string": "".join(term_list),
            "coeff": -1.0,
            "type": "Bulk Term",
            "indices": (j - 1, j, j + 1)
        })

    # Add boundary term 2: -Z_{n-2} X_{n-1}
    term_list_b2 = ['I'] * n_qubit
    term_list_b2[n_qubit - 2] = 'Z'
    term_list_b2[n_qubit - 1] = 'X'
    hamiltonian_terms.append({
        "string": "".join(term_list_b2),
        "coeff": -1.0,
        "type": "Boundary Term",
        "indices": (n_qubit - 2, n_qubit - 1)
    })

    # --- Printing Section: Output sorted by position ---
    # Use a lambda function to sort terms based on the index of the first qubit they act on.
    sorted_terms_for_printing = sorted(hamiltonian_terms, key=lambda term: term["indices"][0])

    print("Pauli terms defined by strings (sorted by physical location):")
    for term in sorted_terms_for_printing:
        p_str = term["string"]
        term_type = term["type"]
        indices_str = ", ".join(map(str, term["indices"]))
        print(f"- {p_str}  ({term_type}, acting on qubits {indices_str})")

    # --- Calculation Section: Use the original (or any order) list ---
    # Extract the required Pauli strings and coefficients for calculation from the hamiltonian_terms list.
    pauli_strings_str = [term["string"] for term in hamiltonian_terms]
    coeffs = [term["coeff"] for term in hamiltonian_terms]

    # 1. Convert the list of strings to a numerical list.
    pauli_terms_numerical = convert_pauli_strings_to_numerical(pauli_strings_str)

    # 2. Convert the numerical list to a TensorFlow tensor.
    pauli_terms_tf = tf.constant(pauli_terms_numerical, dtype=tf.int32)
    coeffs_tf = tf.constant(coeffs, dtype=tf.complex128)

    # 3. Build the Hamiltonian matrix using the TensorCircuit function.
    hamiltonian_cluster_model = tc.quantum.PauliStringSum2Dense(pauli_terms_tf, coeffs_tf)

    print("\nSuccessfully constructed the Hamiltonian for the Transverse Field Cluster Model.")
    print("Shape of the Hamiltonian matrix (dense form):", hamiltonian_cluster_model.shape)

    # -------------------------------------------------------------------------

    noise_param = None
    if args.noise:
        print("\nEnabling noise model.")
        noise_param = {'two_qubit_channel_depolarizing_p': args.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': args.single_qubit_depolarizing_p,
                       'bit_flip_p': args.bit_flip_p}

    trainer = VqeTrainerNew(n_cir_parallel=10, n_runs=10, max_iteration=2000,
                            n_qubit=n_qubit, hamiltonian=hamiltonian_cluster_model,
                            noise_param=noise_param)

    run_id = args.seed
    trainer.batch_train_parallel(device_name, task_name, run_id)