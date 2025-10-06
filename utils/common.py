import torch
import os
import pickle
import numpy as np
import networkx as nx
import pennylane as qml
import matplotlib.pyplot as plt


def load_pkl(input_file):
    """Load a .pkl file."""
    with open(input_file, 'rb') as f:
        output_file = pickle.load(f)
    return output_file


def save_pkl(data, loc):
    """
    Save data to a .pkl file.
    :param data: The data to be saved.
    :param loc: The file path.
    """
    with open(loc, 'wb') as f:
        pickle.dump(data, file=f)
    return 0


def format_execution_time(execution_time):
    """
    Formats the execution time into a human-readable string.
    :param execution_time: Time in seconds.
    :return: Formatted string representing the time.
    """
    if execution_time < 1e-6:
        return f"{execution_time * 1e9:.2f} ns"
    elif execution_time < 1e-3:
        return f"{execution_time * 1e6:.2f} μs"
    elif execution_time < 1:
        return f"{execution_time * 1e3:.2f} ms"
    elif execution_time < 60:
        return f"{execution_time:.2f} s"
    elif execution_time < 3600:
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        return f"{minutes} min {seconds:.2f} s"
    elif execution_time < 86400:
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        return f"{hours} h {minutes} min {seconds:.2f} s"
    else:
        days = int(execution_time // 86400)
        hours = int((execution_time % 86400) // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        return f"{days} days {hours} h {minutes} min {seconds:.2f} s"


def ensure_dir(path):
    """
    Ensures that a directory exists. If it doesn't, it creates one.
    :param path: The directory path.
    :return: True if the directory was created, False otherwise.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def get_torch_device():
    """
    Get the appropriate PyTorch device (CUDA if available, otherwise CPU).
    :return: torch.device: The PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # Uncomment the following lines if you want to use MPS on Apple Silicon
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


def make_it_unique(circuit, num_qubit):
    '''
    Rearranges a circuit by moving all gates to the far left (as early as possible).
    :param circuit: A circuit in list format, e.g., [['rz', 0], ['cz', 2, 3], ['ry', 1]].
    :return:
        - list: A rearranged gate sequence.
        - int: Circuit depth.
        - int: Total number of gates.
        - int: Number of 2-qubit gates.
    '''
    bitmap = [[] for _ in range(num_qubit)]

    num_1_qb_gate, num_2_qb_gate = 0, 0
    for gate in circuit:
        if len(gate) == 2:  # 1-qubit gate
            bitmap[gate[1]].append(gate)
            num_1_qb_gate += 1
        else:  # 2-qubit gate
            qb_long, qb_short = (gate[1], gate[2]) if len(bitmap[gate[1]]) >= len(bitmap[gate[2]]) else (gate[2],
                                                                                                         gate[1])

            while len(bitmap[qb_short]) < len(bitmap[qb_long]):
                bitmap[qb_short].append(None)

            qb_low, qb_high = (gate[2], gate[1]) if gate[1] > gate[2] else (gate[1], gate[2])

            bitmap[qb_low].append(gate)
            bitmap[qb_high].append(None)
            num_2_qb_gate += 1

    num_bit = [len(b) for b in bitmap]
    depth = max(num_bit) if num_bit else 0  # Depth of the circuit

    uni_cir = []
    for i in range(depth):
        for j in range(num_qubit):
            if num_bit[j] > i and bitmap[j][i] is not None:
                uni_cir.append(bitmap[j][i])

    return uni_cir.copy(), depth, num_1_qb_gate + num_2_qb_gate, num_2_qb_gate


def circuit_list_to_adj(circuit, num_qubit, gate_type_list, require_feat_matrix=False):
    '''
    Produce the adjacency and feature matrices based on the DAG of a circuit.
    :param circuit: list, e.g., [['rz', 0], ['ry', 2], ['cz', 1, 2]], processed by make_it_unique.
    :param num_qubit: int.
    :param gate_type_list: list, e.g., ['rz', 'ry', 'cz'].
    :param require_feat_matrix: bool, whether to return the feature matrix.
    :return: ndarray (adjacency matrix), and optionally ndarray (feature matrix).
    '''
    graph = nx.DiGraph()

    # Add nodes to the graph
    graph.add_node('start')
    for j in range(1, len(circuit) + 1):
        graph.add_node(j)
    graph.add_node('end')

    # Add edges to the graph based on qubit dependencies
    last = ['start' for _ in range(num_qubit)]
    for k, gate in enumerate(circuit):
        node_idx = k + 1
        if len(gate) == 2:  # 1-qubit gate
            graph.add_edge(last[gate[1]], node_idx)
            last[gate[1]] = node_idx
        else:  # 2-qubit gate
            graph.add_edge(last[gate[1]], node_idx)
            graph.add_edge(last[gate[2]], node_idx)
            last[gate[1]] = node_idx
            last[gate[2]] = node_idx

    for k in last:
        graph.add_edge(k, 'end')

    # Get the adjacency matrix
    adj_matrix = nx.adjacency_matrix(graph).todense()

    if require_feat_matrix:
        feat_matrix = []
        for node in graph.nodes:
            # One-hot encoding for gate type: [start, gate_types..., end]
            t1 = [0] * (len(gate_type_list) + 2)
            # One-hot encoding for qubit location
            t2 = [0] * num_qubit

            if node == 'start':
                t1[0] = 1
                t2 = [1] * num_qubit  # Represents availability
            elif node == 'end':
                t1[-1] = 1
                t2 = [1] * num_qubit  # Represents availability
            else:
                gate_info = circuit[node - 1]
                t1[gate_type_list.index(gate_info[0]) + 1] = 1
                t2[gate_info[1]] = 1
                if len(gate_info) == 3:  # 2-qubit gate
                    t2[gate_info[2]] = 1

            t1.extend(t2)
            feat_matrix.append(t1)
        feat_matrix = np.array(feat_matrix)
        return adj_matrix, feat_matrix
    else:
        return adj_matrix


def draw_circuit(cir_lst, num_qubit, dir_file):
    '''
    Draws a quantum circuit and saves it to a file.
    :param cir_lst: list, e.g., [['rz',0], ['ry',1], ['cz',0,1]].
    :param num_qubit: int, number of qubits.
    :param dir_file: str, file path to save the image, e.g., 'result/cir.png'.
    '''
    dev = qml.device('default.qubit', wires=num_qubit)

    @qml.qnode(dev)
    def circuit(cir):
        for gate in cir:
            op_name = gate[0].lower()
            wires = gate[1:]
            if op_name == 'cx':
                qml.CNOT(wires=wires)
            elif op_name == 'cz':
                qml.CZ(wires=wires)
            elif op_name == 'rx':
                qml.RX(0, wires=wires)
            elif op_name == 'ry':
                qml.RY(0, wires=wires)
            elif op_name == 'rz':
                qml.RZ(0, wires=wires)
            elif op_name == 'x':
                qml.PauliX(wires=wires)
            elif op_name == 'sx':
                qml.SX(wires=wires)
            else:
                print(f'Error: Undefined gate "{gate[0]}" in drawing function.')
                exit(123)
        return qml.state()

    qml.draw_mpl(circuit)(cir_lst)
    make_dir(dir_file)
    plt.savefig(dir_file)
    plt.close()


def levenshtein_distance(s1, s2):
    '''
    Calculates the Levenshtein distance between two sequences.
    :param s1: list, a sequence.
    :param s2: list, another sequence.
    :return: int, the Levenshtein distance.
    '''
    m, n = len(s1), len(s2)
    # Create an (m+1)x(n+1) matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize boundary conditions
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                           dp[i][j - 1] + 1,  # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution

    return dp[m][n]


def make_dir(dir_file):
    '''
    Ensures the directory for a given file path exists.
    :param dir_file: e.g., 'result/1.txt'.
    '''
    directory = os.path.dirname(dir_file)  # e.g., 'result'
    if directory:
        os.makedirs(directory, exist_ok=True)