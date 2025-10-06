from qiskit import transpile
from qiskit import QuantumCircuit
import utils
import numpy as np


def optimize(cir, gate_set, num_qubit):
    '''
    :param cir: a circuit in list format, e.g., [['rz',0],['ry',1],['cz',1,2]]
    :param gate_set: the native gate set, e.g., ['cx', 'rz', 'ry']
    :return: [lst, depth, num_gate, num_2_qb_gate]
            lst is the optimized circuit in list format
    '''

    cir_qk = list_to_qiskit(cir, num_qubit)
    cir_qk = transpile(cir_qk, optimization_level=1, basis_gates=gate_set)
    lst = qiskit_to_list(cir_qk)

    lst, depth, num_gate, num_2_qb_gate = utils.make_it_unique(lst, num_qubit=num_qubit)

    return [lst, depth, num_gate, num_2_qb_gate]

def list_to_qiskit(cir, num_qubit):
    '''
    :param cir: a circuit in list format, e.g., [['rz',0],['ry',1],['cz',1,2]]
    :return: a circuit in qiskit format
    '''
    cir_qk = QuantumCircuit(num_qubit)
    for gate in cir:
        if gate[0] == 'cx':
            cir_qk.cx(gate[1], gate[2])
        elif gate[0] == 'cz':
            cir_qk.cz(gate[1], gate[2])
        elif gate[0] == 'rx':
            cir_qk.rx(theta=np.random.normal(), qubit=gate[1])
        elif gate[0] == 'ry':
            cir_qk.ry(theta=np.random.normal(), qubit=gate[1])
        elif gate[0] == 'rz':
            cir_qk.rz(phi=np.random.normal(), qubit=gate[1])
        elif gate[0] == 'x':
            cir_qk.x(gate[1])
        elif gate[0] == 'sx':
            cir_qk.sx(gate[1])
        else:
            print('undefined gate name found in function list_to_qiskit.')
            exit(123)
    return cir_qk

def qiskit_to_list(cir):
    '''
    :param cir: a circuit in qiskit format
    :return: a circuit in list format, e.g., [['rz',0], ['ry',1], ['cz',0,1]]
    '''

    cir_list = []

    for operation, qubits, _ in cir.data:    # iterate all gates
        gate_list = []
        gate_list.append(operation.name)
        for q in qubits:
            gate_list.append(q._index)
        cir_list.append(gate_list)

    return cir_list

if __name__ == "__main__":
    # for test
    cir_qk = QuantumCircuit(7)
    lst = qiskit_to_list(cir_qk)
    lst, depth, num_gate, num_2_qb_gate = utils.make_it_unique(lst, num_qubit=7)
    print(lst, depth, num_gate, num_2_qb_gate)

