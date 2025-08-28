from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
import numpy as np

def get_evolution_bloch_vectors(circuit):
    """Return Bloch vectors after each gate in the circuit."""
    num_qubits = circuit.num_qubits
    sv = Statevector.from_label('0' * num_qubits)

    bloch_snapshots = []

    for idx, (instr, qargs, _) in enumerate(circuit.data):
        sv = sv.evolve(instr)
        rho = DensityMatrix(sv)

        snapshot = []
        for qubit in range(num_qubits):
            reduced = partial_trace(rho, [i for i in range(num_qubits) if i != qubit])
            rho_single = np.array(reduced.data)

            sigma_x = np.array([[0, 1], [1, 0]])
            sigma_y = np.array([[0, -1j], [1j, 0]])
            sigma_z = np.array([[1, 0], [0, -1]])

            x = np.real(np.trace(rho_single @ sigma_x))
            y = np.real(np.trace(rho_single @ sigma_y))
            z = np.real(np.trace(rho_single @ sigma_z))

            snapshot.append({"qubit": qubit, "x": x, "y": y, "z": z})
        bloch_snapshots.append({"step": idx+1, "vectors": snapshot})

    return bloch_snapshots
