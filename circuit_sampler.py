import itertools
import random
import numpy as np
import optimize_circuit
import config
import networkx as nx

def get_maximal_valid_matchings(connection):
    edge_set = connection


    def is_valid_matching(edge_subset):
        used_nodes = set()
        for a, b in edge_subset:
            if a in used_nodes or b in used_nodes:
                return False
            used_nodes.update([a, b])
        return True


    def find_all_valid_matchings(edges):
        all_valid = []
        for r in range(1, len(edges) + 1):
            for subset in itertools.combinations(edges, r):
                if is_valid_matching(subset):
                    sorted_subset = sorted([tuple(sorted(pair)) for pair in subset])
                    if sorted_subset not in all_valid:
                        all_valid.append(sorted_subset)
        return all_valid

    def remove_subset_combinations(combos):
        combos_set = [set(map(tuple, combo)) for combo in combos]
        filtered = []
        for i, current in enumerate(combos_set):
            is_subset = False
            for j, other in enumerate(combos_set):
                if i != j and current.issubset(other):
                    is_subset = True
                    break
            if not is_subset:
                filtered.append(combos[i])
        return filtered

    all_valid_combos = find_all_valid_matchings(edge_set)
    max_len = max(len(c) for c in all_valid_combos)
    min_len = max_len - 1

    filtered_by_len = [c for c in all_valid_combos if len(c) >= min_len]

    final_combos = remove_subset_combinations(filtered_by_len)

    return final_combos


class CircuitSampler:
    def __init__(self, device_name, num_layer, num_qubit):
        '''
        :param device_name: e.g., 'grid'
        '''
        self.device = config.device[device_name]
        self.n_qubit = num_qubit
        self.num_layer = num_layer

    def is_connected(self, graph):
        """Check whether the graph is connected"""
        return nx.is_connected(graph)

    def generate_connected_subgraph(self, graph, num_nodes):
        """
        From the given graph try to generate a connected subgraph with a specified number of nodes.
        graph (NetworkX graph): The given graph.
        num_nodes (int): The number of nodes in the desired connected subgraph.
        Returns:
            subgraph (NetworkX graph): This is a connected subgraph with the specified number of nodes.
            subgraph_nodes (list): This is a list of nodes in the subgraph.
        """
        if num_nodes > graph.number_of_nodes():
            return graph, list(graph.nodes())

        while True:
            nodes = list(graph.nodes())
            selected_nodes = np.random.choice(nodes, num_nodes, replace=False)
            subgraph = graph.subgraph(selected_nodes)
            if self.is_connected(subgraph):
                return subgraph, selected_nodes.tolist()

    def generate_qubit_mappings(self):
        """
        quantum_device_connection: This is a list of connection relationships for the quantum device
        qubit_num: The number of logical bits
        """
        quantum_device_connection = self.device['connectivity']
        device_graph = nx.Graph(quantum_device_connection)
        subgraph, subgraph_nodes = self.generate_connected_subgraph(device_graph, self.n_qubit)
        physical_qubits = subgraph_nodes
        mapping = {physical_qubits[i]: i for i in range(self.n_qubit)}
        logical_connections = [np.sort([mapping[edge[0]], mapping[edge[1]]]) for edge in subgraph.edges()]
        return mapping, logical_connections

    def sample(self, num_sample, result=None, qubit_mapping_lst=None, edges_mapping_list=None):

        if result == None:
            result = []
            qubit_mapping_lst = []
            edges_mapping_list = []

        sample_count = 0
        while sample_count < num_sample:
            cir, qubit_mapping, edges_mapping = self.generateCircuit()
            opt_cir = optimize_circuit.optimize(cir, self.device['gate_set'], num_qubit=self.n_qubit)    # call transpile function to optimize
            if opt_cir[1] == 0:    # depth is zero
                continue
            # whether the transpiled circuit still satisfies the connectivity
            if not self.check_connectivity(opt_cir[0], qubit_mapping):
                exit(123)
            if opt_cir not in result:
                result.append(opt_cir)
                edges_mapping_list.append(edges_mapping)
                qubit_mapping_lst.append(qubit_mapping)
                # phy2log_lst.append(phy2log)
                sample_count = sample_count + 1
                if sample_count % 1000 == 0:
                    print(sample_count)

        return result, qubit_mapping_lst, edges_mapping_list

    def generateCircuit(self):
        '''
        using layer-wise pipeline to generate a logical circuit in list format e.g., [['rz',0],['ry',1],['cz',1,2]]
        return the logical circuit, and the phy2log mapping
        '''

        # generate a random mapping from physical to logical qubits
        qubit_mapping, edges_mapping = self.generate_qubit_mappings()
        layout = get_maximal_valid_matchings(edges_mapping)

        # generate a distribution of gate types for a circuit
        logit_gate_type = np.random.normal(size=len(self.device['gate_set']))
        prob_gate_type = np.exp(logit_gate_type) / sum(np.exp(logit_gate_type))

        # layer-wise generation
        cir = []
        for _ in range(self.num_layer):
            gate_type = np.random.choice(self.device['gate_set'], p=prob_gate_type)
            if gate_type in ['cz', 'cx']:
                layout_idx = np.random.randint(0, len(layout))
                selected_layout = layout[layout_idx]
                for i in range(len(selected_layout)):
                    q0 = selected_layout[i][0]
                    q1 = selected_layout[i][1]
                    if gate_type == 'cx' and random.choice([0, 1]) == 1:
                        q0, q1 = q1, q0
                    cir.append([gate_type, q0, q1])
            else:
                for i in range(self.n_qubit):
                    cir.append([gate_type, i])

        return cir, qubit_mapping, edges_mapping



    def check_connectivity(self, cir,  phy_to_log):
        '''
        :param cir: e.g., [['rz',0],['ry',1],['cz',1,2]]
        :param phy2log: e.g., {2: 6, 3: 3, 5: 0, 6: 4, 7: 2, 9: 1, 10: 5, 13: 7} phy: log
        :return: true if the circuit satisfy the connectivity
        '''
        log_to_phy = {v: k for k, v in phy_to_log.items()}
        for gate in cir:
            if len(gate) == 3:
                a = log_to_phy[gate[1]]
                b = log_to_phy[gate[2]]
                c, d = min(a, b), max(a, b)
                if [c, d] not in self.device['connectivity']:
                    print('transpile violates the connectivity')
                    return False
        return True

