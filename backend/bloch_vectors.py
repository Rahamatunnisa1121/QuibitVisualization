from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
import numpy as np

def get_single_qubit_bloch_vectors(circuit: QuantumCircuit):
    """
    Given a quantum circuit, return Bloch vectors for each qubit.
    Output: list of dicts [{qubit: i, x:..., y:..., z:...}, ...]
    """
    state = Statevector.from_instruction(circuit)
    rho = DensityMatrix(state)

    bloch_vectors = []
    num_qubits = circuit.num_qubits

    for qubit in range(num_qubits):
        reduced_state = partial_trace(rho, [i for i in range(num_qubits) if i != qubit])
        rho_single = np.array(reduced_state.data)

        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])

        x = np.real(np.trace(rho_single @ sigma_x))
        y = np.real(np.trace(rho_single @ sigma_y))
        z = np.real(np.trace(rho_single @ sigma_z))

        bloch_vectors.append({"qubit": qubit, "x": x, "y": y, "z": z})

    return bloch_vectors
