import argparse
import time
import os
import numpy as np
import tensorcircuit as tc
import tensorflow as tf
from utils.common import load_pkl, save_pkl
from multiprocessing import Pool
import matplotlib.pyplot as plt

# --- Global Configurations ---
tc.set_dtype("complex128")
tc.set_backend("tensorflow")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only

from quantum_gates import Gate


def convert_gate_list(raw_gate_list):
    """Converts a raw list of gates into a list of Gate objects."""
    gate_objects = []
    for item in raw_gate_list:
        name = item[0]
        # Standardize gate names
        name = {'ry': 'Ry', 'rz': 'Rz', 'rx': 'Rx', 'cz': 'CZ', 'cx': 'CNOT'}.get(name, name)
        qubit_indices = item[1:]
        qubits = len(qubit_indices)
        para_gate = (name in ['Rz', 'Ry', 'Rx'])
        gate = Gate(qubits=qubits, name=name, para_gate=para_gate)
        gate.act_on = qubit_indices
        gate_objects.append(gate)
    return gate_objects


def get_param_gate_count(circuits):
    """
    Counts the number of parameterized gates in each circuit.
    :param circuits: A list of circuits, where each circuit is a list of Gate objects.
    :return: A numpy array containing the parameterized gate count for each circuit.
    """
    cir_param_gate_count = []
    for circuit in circuits:
        param_gate_count = 0
        for gate in circuit:
            if gate.para_gate:
                param_gate_count += 1
        cir_param_gate_count.append(param_gate_count)
    return np.array(cir_param_gate_count)


class RFCalculator:
    """
    A class to calculate the Reachability Factor (RF) for quantum circuits.
    """

    def __init__(self, args, hamiltonian, noise_param=None):
        self.args = args
        self.n_qubit = args.qubits
        self.parallel = args.parallel
        self.seed = args.seed
        self.sample_num = args.sample_lines  # Number of circuits to sample
        self.theta_num = args.theta_num  # Number of parameter sets to sample per circuit

        self.K = tc.set_backend("tensorflow")

        # Noise-related parameters. Ignored if noise is False.
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

        # Hamiltonian
        self.hamiltonian_ = hamiltonian
        self.lattice = tc.templates.graphs.Line1D(self.n_qubit, pbc=self.hamiltonian_['pbc'])
        self.h = tc.quantum.heisenberg_hamiltonian(self.lattice, hzz=self.hamiltonian_['hzz'],
                                                   hxx=self.hamiltonian_['hxx'], hyy=self.hamiltonian_['hyy'],
                                                   hx=self.hamiltonian_['hx'], hy=self.hamiltonian_['hy'],
                                                   hz=self.hamiltonian_['hz'], sparse=self.hamiltonian_['sparse'])

    def compute_energy(self, param, structure):
        """
        Computes the energy expectation for a given circuit structure and parameters.
        :param param: Circuit parameters.
        :param structure: The circuit structure (list of Gate objects).
        :return: The calculated energy (expectation value).
        """
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            c = tc.DMCircuit(self.n_qubit)
            for idx, gate in enumerate(structure):
                if gate.name == "CX":
                    c.cx(gate.act_on[0], gate.act_on[1])
                    c.general_kraus(self.two_qubit_dep_channel, gate.act_on[0], gate.act_on[1])
                elif gate.name == "CZ":
                    c.cz(gate.act_on[0], gate.act_on[1])
                    c.general_kraus(self.two_qubit_dep_channel, gate.act_on[0], gate.act_on[1])
                elif gate.name == "Ry":
                    c.ry(gate.act_on[0], theta=param[idx])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                elif gate.name == "Rz":
                    c.rz(gate.act_on[0], theta=param[idx])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                elif gate.name == "Rx":
                    c.rx(gate.act_on[0], theta=param[idx])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                else:
                    raise ValueError(f"Invalid gate name: {gate.name}")

            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            # Calculate energy
            st = c.state()
            x = tf.matmul(st, self.h)
            e = tf.linalg.trace(x)
            e = self.K.real(e)

        else:  # Noiseless simulation
            c = tc.Circuit(self.n_qubit)
            for idx, gate in enumerate(structure):
                if gate.name == "CX":
                    c.cx(gate.act_on[0], gate.act_on[1])
                elif gate.name == "CZ":
                    c.cz(gate.act_on[0], gate.act_on[1])
                elif gate.name == "Ry":
                    c.ry(gate.act_on[0], theta=param[idx])
                elif gate.name == "Rz":
                    c.rz(gate.act_on[0], theta=param[idx])
                elif gate.name == "Rx":
                    c.rx(gate.act_on[0], theta=param[idx])
                else:
                    raise ValueError(f"Invalid gate name: {gate.name}")
            e = tc.templates.measurements.operator_expectation(c, self.h)

        return e

    def get_parallel(self):
        """Gets a JIT-compiled function for parallel energy and gradient computation."""
        parallel = tc.backend.value_and_grad(self.compute_energy, argnums=0)
        parallel = tc.backend.jit(parallel)
        return parallel

    def train_circuit(self, work_queue):
        """
        Calculates energies for a single circuit with multiple random parameter sets.
        This function is designed to be called by a multiprocessing Pool.
        """
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        tc.set_backend("tensorflow")
        tc.set_dtype("complex128")
        parallel = self.get_parallel()
        sample = work_queue[0]

        energies = []
        for i in range(self.theta_num):
            par = np.random.uniform(0, 1, len(sample)) * np.pi * 2
            param = tf.Variable(
                initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
            )
            energy, grad = parallel(param, sample)
            energies.append(energy.numpy())

        return energies

    def process(self, circuits):
        """
        Calculates the cost (loss) for a list of circuits in parallel.
        :param circuits: A list of circuit structures.
        :return: A numpy array of costs, shape (num_circuits, num_thetas).
        """
        work_queue = []
        for i in range(len(circuits)):
            work_queue.append([circuits[i]])

        pool = Pool(processes=self.parallel)
        results = pool.map(self.train_circuit, work_queue)
        pool.close()
        pool.join()

        loss = np.array(results)  # shape (num_samples, num_thetas)
        return loss

    def calculate_RF(self, circuits):
        """Calculates the Reachability Factor (RF) for the given circuits."""
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # Calculate Cost, i.e., Loss
        cost = self.process(circuits)

        # Calculate RF
        # Variance
        var = np.var(cost, axis=1)  # shape (num_samples,)
        var_sqrt = np.sqrt(var)  # shape (num_samples,)
        lambda_1_norm = 32
        print("Denominator for sigma (lambda_1_norm): ", lambda_1_norm)
        sigma = var_sqrt / lambda_1_norm  # shape (num_samples,)

        M = get_param_gate_count(circuits)
        # The result of rf1 is equivalent to rf, so either can be used.
        RF = np.sqrt(2 * M) * sigma
        return RF


def main(args):
    """1. Sample N circuits, default is 100,000."""
    cir_path = f"data/raw/grid_16q_Heisenberg_8/pretrain/run_{args.seed}/samples.pkl"
    samples = load_pkl(cir_path)
    samples = samples[:args.sample_lines]
    list_cir = [data[0] for data in samples]
    circuits = [convert_gate_list(cir) for cir in list_cir]
    print("Circuits loaded from file, count:", len(circuits))

    # Noise information
    noise_param = None
    if args.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': 0.01,
                       'single_qubit_channel_depolarizing_p': 0.001,
                       'bit_flip_p': 0.01}

    # Hamiltonian
    hamiltonian = {'pbc': True, 'hzz': 1, 'hxx': 1, 'hyy': 1, 'hx': 0, 'hy': 0, 'hz': 1, 'sparse': False}

    """2. Calculate the RF for the circuits."""
    RF_calculator = RFCalculator(args, hamiltonian, noise_param=noise_param)
    # Calculate RF
    RF_result = RF_calculator.calculate_RF(circuits)
    print("RF calculation complete, number of results:", len(RF_result))

    """3. Record the calculation results."""
    # Record RF results
    result_dir = f"data/raw/grid_16q_Heisenberg_8/pretrain/run_{args.seed}/"
    RF_result_path = os.path.join(result_dir, f"RF_result_samplelines{args.sample_lines}_noise.pkl")
    save_pkl(RF_result, RF_result_path)
    print("RF results saved to:", RF_result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to calculate Reachability Factor for quantum circuits.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    # Metric information
    parser.add_argument("--proxy", type=str, default='RF', help="Proxy metric name.")
    # Task type: Heisenberg
    parser.add_argument("--task", type=str, default='Heisenberg_8', help="Task name.")
    # Search space
    parser.add_argument('--search_space', type=str, default='layer', help='Search space identifier.')
    # Number of parallel processes
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel processes.")
    # Number of qubits for the task
    parser.add_argument("--qubits", type=int, default=8, help="Number of qubits.")
    # Add parameter for number of sampled circuits
    parser.add_argument("--sample_lines", type=int, default=100, help="Number of sampled circuits.")
    # Add parameter for number of thetas
    parser.add_argument("--theta_num", type=int, default=1000, help="Number of parameter sets per circuit.")
    # Noise information
    parser.add_argument("--noise", type=bool, default=True, help="Enable/disable noise simulation.")
    parser.add_argument("--two_qubit_channel_depolarizing_p", type=float, default=0.01,
                        help="Two-qubit depolarizing noise level.")
    parser.add_argument("--single_qubit_channel_depolarizing_p", type=float, default=0.001,
                        help="Single-qubit depolarizing noise level.")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="Measurement bit-flip noise level.")
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f'Total run time: {end_time - start_time:.2f}s')
    print('------------------------')