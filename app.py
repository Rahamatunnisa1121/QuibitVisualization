import streamlit as st
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
import numpy as np
import json
import os
from qiskit import ClassicalRegister
from utils.ai_validator import CircuitValidator
from qiskit import QuantumCircuit

from utils.ai_validator import CircuitValidator
from qiskit import QuantumCircuit

# Import Qiskit equivalent circuits
from examples.circuits_equivalent import (
    hadamard_circuit,
    bell_state_circuit,
    ghz_state_circuit,
    qft_circuit,
    grover_circuit
)


# ---- Backend Function ---- #
def get_single_qubit_bloch_vectors(circuit: QuantumCircuit):
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


# ---- Plot Bloch Sphere ---- #
def plot_bloch_sphere(bloch_vector, qubit_index):
    x, y, z = bloch_vector["x"], bloch_vector["y"], bloch_vector["z"]

    fig = go.Figure()

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    fig.add_surface(x=xs, y=ys, z=zs, opacity=0.1, colorscale="Blues")

    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode="lines+markers",
        line=dict(color="red", width=5),
        marker=dict(size=4, color="red")
    ))

    fig.update_layout(
        title=f"Qubit {qubit_index} Bloch Vector",
        scene=dict(
            xaxis=dict(range=[-1,1]),
            yaxis=dict(range=[-1,1]),
            zaxis=dict(range=[-1,1])
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig


# ---- Load Circuit from JSON ---- #

def load_circuit_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    qc = QuantumCircuit(data["qubits"])

    for op in data["operations"]:
        gate = op["gate"]
        targets = op["targets"]
        params = op.get("params", {})

        if gate == "h":
            qc.h(*targets)

        elif gate == "x":
            qc.x(*targets)

        elif gate == "cx":
            qc.cx(*targets)

        elif gate == "cz":
            qc.cz(*targets)

        elif gate == "swap":
            qc.swap(*targets)

        elif gate == "cp":
            theta = params.get("theta", np.pi/2)  # default to π/2 if not given
            qc.cp(theta, *targets)

        elif gate == "measure":
            if qc.num_clbits < len(targets):
                qc.add_register(ClassicalRegister(len(targets)))
            qc.measure(*targets, *targets)

        else:
            raise ValueError(f"❌ Unsupported gate: {gate}")

    return qc


# ---- Streamlit UI ---- #
st.title("🔮 Quantum State Visualizer")
st.write("Visualize quantum circuits on the Bloch sphere in real time.")

mode = st.radio("Choose mode:", ["Predefined Circuits", "JSON Circuits", "Custom Code"])

# --- Predefined Circuits (Python functions) --- #
if mode == "Predefined Circuits":
    example = st.selectbox("Choose circuit:", ["Hadamard", "Bell State", "GHZ"])

    if example == "Hadamard":
        qc = hadamard_circuit()
    elif example == "Bell State":
        qc = bell_state_circuit()
    elif example == "GHZ":
        qc = ghz_state_circuit()

# --- JSON Circuits (load from /examples) --- #
elif mode == "JSON Circuits":
    json_files = [f for f in os.listdir("examples") if f.endswith(".json")]
    choice = st.selectbox("Choose JSON circuit:", json_files)

    path = os.path.join("examples", choice)
    qc = load_circuit_from_json(path)

# --- Custom Code Input --- #
elif mode == "Custom Code":
    from qiskit import QuantumCircuit
    from utils.ai_validator import CircuitValidator

    st.subheader("📝 Enter Qiskit Circuit Code")
    user_code = st.text_area(
        "Write your circuit here (use variable name qc):",
        height=200,
        placeholder="Example:\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)"
    )

    if user_code.strip():
        try:
            local_vars = {}
            exec(user_code, {}, local_vars)

            if "qc" in local_vars:
                qc = local_vars["qc"]

                # ✅ Use AI Validator
                is_valid, message = CircuitValidator.validate_qiskit_circuit(qc)

                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
                    qc = QuantumCircuit(1)
                    qc.h(0)  # fallback

            else:
                st.error("❌ No circuit named qc found. Please define qc in your code.")
                qc = QuantumCircuit(1)
                qc.h(0)

        except Exception as e:
            st.error(f"❌ Error in your code: {e}")
            qc = QuantumCircuit(1)
            qc.h(0)

    else:
        st.info("ℹ No circuit provided, showing fallback Hadamard.")
        qc = QuantumCircuit(1)
        qc.h(0)



# ---- Compute & Display ---- #
bloch_vectors = get_single_qubit_bloch_vectors(qc)

st.subheader("Quantum Circuit")
st.subheader("Bloch Spheres")

# --- AI-Powered Circuit Optimization --- #
st.subheader("AI-Powered Circuit Optimization")

if st.button("Optimize Circuit"):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    optimized_qc = transpile(qc, optimization_level=3)
    st.write("**Original Circuit:**")
    st.text(qc.draw())
    st.write(f"Gate count: {qc.size()}, Depth: {qc.depth()}")
    st.write("**Optimized Circuit:**")
    st.text(optimized_qc.draw())
    st.write(f"Gate count: {optimized_qc.size()}, Depth: {optimized_qc.depth()}")
    if optimized_qc.size() < qc.size():
        st.success(f"✅ Optimization reduced gate count by {qc.size() - optimized_qc.size()} gates!")
    else:
        st.info("ℹ️ No further optimization possible.")

    # Validate equivalence of quantum states
    try:
        sv_orig = Statevector.from_instruction(qc)
        sv_opt = Statevector.from_instruction(optimized_qc)
        if sv_orig.equiv(sv_opt):
            st.success("✅ Optimized circuit is functionally equivalent to the original.")
        else:
            st.warning("⚠️ Optimized circuit produces a different quantum state.")
    except Exception as e:
        st.error(f"Error during validation: {e}")

    # Use optimized circuit for Bloch visualization
    bloch_vectors = get_single_qubit_bloch_vectors(optimized_qc)
else:
    st.text(qc.draw())
    st.write(f"Gate count: {qc.size()}, Depth: {qc.depth()}")
    # Use original circuit for Bloch visualization
    bloch_vectors = get_single_qubit_bloch_vectors(qc)

st.subheader("Bloch Spheres")
for vec in bloch_vectors:
    fig = plot_bloch_sphere(vec, vec["qubit"])
    st.plotly_chart(fig, use_container_width=True)