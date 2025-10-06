import os
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import utils.common as utils
import argparse
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
        self.hamiltonian_ = hamiltonian
        self.lattice = tc.templates.graphs.Line1D(self.n_qubit, pbc=self.hamiltonian_['pbc'])
        self.h = tc.quantum.heisenberg_hamiltonian(self.lattice, hzz=self.hamiltonian_['hzz'],
                                                   hxx=self.hamiltonian_['hxx'], hyy=self.hamiltonian_['hyy'],
                                                   hx=self.hamiltonian_['hx'], hy=self.hamiltonian_['hy'],
                                                   hz=self.hamiltonian_['hz'], sparse=self.hamiltonian_['sparse'])
        self.give_up_rest = False
        self.solution = None

        """ Noise-related parameter, don't care if noise is False. """
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
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.two_qubit_channel_depolarizing_p/15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.single_qubit_channel_depolarizing_p/3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

    def compute_energy(self, param, structure):
        """
        :param param: Circuit Parameters
        :param structure: Circuit
        :return:
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
            if cir[i][0] == 'rx':
                param_num += 1
            if cir[i][0] == 'ry':
                param_num += 1
            if cir[i][0] == 'rz':
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
        param_initial = param.numpy()
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
                        # print(distance.max())
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

        # Adjust layout and save the plot
        plt.tight_layout()
        save_path_img = f'{exp_dir}/circuit_train_curve/'
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        # plt.show()
        plt.savefig(save_path_img + f'circuit_{circuit_id}_{best_index}.png')
        plt.close()

    def batch_train_parallel(self, device_name, task_name, run_id):
        start_time = time.time()
        # load finetune circuits
        exp_dir = f'data/raw/{device_name}_{task_name}/training/run_{run_id}/'
        samples = utils.load_pkl(f'{exp_dir}samples.pkl')[:500]

        # start training
        work_queue = []
        for i in range(0, len(samples)):
            print(f'circuit id: {i}')
            # load finetune circuits
            work_queue.extend([[samples[i], j] for j in range(0, self.n_runs)])

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
        for i in range(0, len(samples)):
            index0 = i * self.n_runs
            index1 = index0 + self.n_runs
            best_index = np.argmin(energy[index0:index1])
            best_index = best_index + index0
            energy_f.append(energy[best_index])
            param_f.append(param[best_index])
            energy_epoch_f.append(energy_epoch[best_index])
            # self.draw(energy_epoch[best_index], exp_dir, i, best_index)

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        utils.save_pkl(energy_f, f'{exp_dir}energy.pkl')
        utils.save_pkl(param_f, f'{exp_dir}param.pkl')
        utils.save_pkl(energy_epoch_f, f'{exp_dir}energy_epoch.pkl')
        end_time = time.time()
        duration = end_time - start_time
        print(f"run time: {int(duration // 3600)} hours {int((duration % 3600) // 60)} minutes")
        utils.save_pkl(duration, exp_dir + 'duration.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=bool, default=True, help="Whether to consider noise in the simulation")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01, help="Noise level for the two-qubit depolarizing channel")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001, help="Noise level for the single-qubit depolarizing channel")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="Noise level for bit-flip errors")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    device_name = 'grid_16q'    # see config.py
    task_name = 'Heisenberg_8'
    qubit = 8

    noise_param = None
    if args.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': args.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': args.single_qubit_depolarizing_p,
                       'bit_flip_p': args.bit_flip_p}
    hamiltonian = {'pbc': True, 'hzz': 1, 'hxx': 1, 'hyy': 1, 'hx': 0, 'hy': 0, 'hz': 1, 'sparse': False}
    trainer = VqeTrainerNew(n_cir_parallel=10, n_runs=10, max_iteration=2000, n_qubit=qubit, hamiltonian=hamiltonian, noise_param=noise_param)

    run_id = args.seed
    trainer.batch_train_parallel(device_name, task_name, run_id)