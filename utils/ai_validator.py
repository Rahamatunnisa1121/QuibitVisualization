from qiskit import QuantumCircuit
import traceback

class CircuitValidator:

    @staticmethod
    def validate_qiskit_circuit(qc: QuantumCircuit):
        """
        Validate a Qiskit QuantumCircuit object.
        Returns (bool, str) -> (is_valid, message).
        """
        try:
            # --- 1. Basic checks ---
            if not isinstance(qc, QuantumCircuit):
                return False, "Provided object is not a QuantumCircuit."

            if qc.num_qubits == 0:
                return False, "Circuit must have at least one qubit."

            # --- 2. Gate validation ---
            for instr, qargs, cargs in qc.data:
                gate_name = instr.name

                # Example: block invalid gates
                if gate_name in ["czx", "invalid", "fake"]:
                    return False, f"Invalid or unsupported gate: {gate_name}"

                # Prevent measurements if visualizer can't handle classical bits
                if gate_name == "measure":
                    return False, "Measurement operations are not supported in this visualizer."

            # --- 3. If passes all ---
            return True, "Circuit is valid and feasible."

        except Exception as e:
            return False, f"Validation error: {traceback.format_exc()}"