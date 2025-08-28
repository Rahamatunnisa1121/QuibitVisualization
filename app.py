import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
from qiskit_aer import AerSimulator
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from typing import List, Dict, Tuple

# Import predefined circuits (assuming they exist)
# from examples.circuits_equivalent import (
#     hadamard_circuit,
#     bell_state_circuit,
#     ghz_state_circuit,
#     qft_circuit,
#     grover_circuit
# )

# Placeholder functions for the examples
def hadamard_circuit():
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc

def bell_state_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def ghz_state_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc

def qft_circuit():
    qc = QuantumCircuit(3)
    # Simple QFT implementation
    qc.h(0)
    qc.cp(np.pi/2, 0, 1)
    qc.h(1)
    qc.cp(np.pi/4, 0, 2)
    qc.cp(np.pi/2, 1, 2)
    qc.h(2)
    qc.swap(0, 2)
    return qc

def grover_circuit():
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.z([0, 1])
    qc.h([0, 1])
    return qc

# ---------------- Enhanced Utility Functions for 30+ Qubits ---------------- #

def get_sampling_based_expectation_values(qc: QuantumCircuit, selected_qubits: List[int], shots: int = 8192):
    """
    Use sampling to estimate single-qubit expectation values for Pauli operators.
    This approach scales to any number of qubits.
    """
    results = {}
    
    try:
        # Create measurement circuits for each Pauli operator
        for qubit in selected_qubits:
            results[qubit] = {'x': 0, 'y': 0, 'z': 0}
            
            # Z measurement (computational basis)
            qc_z = qc.copy()
            if qc_z.num_clbits == 0:
                qc_z.add_register(ClassicalRegister(1, f'c_z_{qubit}'))
            qc_z.measure(qubit, 0)
            
            backend = AerSimulator()
            job = backend.run(qc_z, shots=shots)
            counts = job.result().get_counts()
            
            # Calculate <Z> expectation value
            p0 = counts.get('0', 0) / shots
            p1 = counts.get('1', 0) / shots
            results[qubit]['z'] = p0 - p1
            
            # X measurement (need to rotate basis)
            qc_x = qc.copy()
            qc_x.ry(-np.pi/2, qubit)  # Rotate Y by -90 degrees to measure X
            if qc_x.num_clbits == 0:
                qc_x.add_register(ClassicalRegister(1, f'c_x_{qubit}'))
            qc_x.measure(qubit, 0)
            
            job = backend.run(qc_x, shots=shots)
            counts = job.result().get_counts()
            
            p0 = counts.get('0', 0) / shots
            p1 = counts.get('1', 0) / shots
            results[qubit]['x'] = p0 - p1
            
            # Y measurement
            qc_y = qc.copy()
            qc_y.rx(np.pi/2, qubit)  # Rotate X by 90 degrees to measure Y
            if qc_y.num_clbits == 0:
                qc_y.add_register(ClassicalRegister(1, f'c_y_{qubit}'))
            qc_y.measure(qubit, 0)
            
            job = backend.run(qc_y, shots=shots)
            counts = job.result().get_counts()
            
            p0 = counts.get('0', 0) / shots
            p1 = counts.get('1', 0) / shots
            results[qubit]['y'] = p0 - p1
            
    except Exception as e:
        st.error(f"❌ Sampling-based computation failed: {e}")
        return {}
    
    return results

def get_stabilizer_tableau(qc: QuantumCircuit):
    """
    For stabilizer circuits, use the stabilizer simulator to get exact results.
    """
    try:
        backend = AerSimulator(method='stabilizer')
        job = backend.run(qc)
        result = job.result()
        
        # Try to get stabilizer state information
        if hasattr(result, 'get_stabilizer'):
            stabilizers = result.get_stabilizer()
            return stabilizers
        else:
            return None
    except Exception as e:
        st.warning(f"Stabilizer simulation failed: {e}")
        return None

def approximate_bloch_vectors_tensor_network(qc: QuantumCircuit, selected_qubits: List[int]):
    """
    Use tensor network approximation for very large circuits.
    This is a placeholder for more advanced tensor network methods.
    """
    # This would typically use libraries like TensorNetwork, Cirq, or custom implementations
    # For now, we'll use a simplified approach with random sampling
    
    results = {}
    for qubit in selected_qubits:
        # Use the sampling approach as fallback
        sampling_results = get_sampling_based_expectation_values(qc, [qubit], shots=4096)
        if qubit in sampling_results:
            results[qubit] = sampling_results[qubit]
    
    return results

def plot_bloch_sphere_enhanced(expectation_values: Dict, title: str = "Bloch Sphere Visualization"):
    """Enhanced Bloch sphere plotting with better visualization."""
    fig = go.Figure()

    # Create a more detailed Bloch sphere
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.1,
        colorscale="Blues",
        showscale=False,
        name="Bloch Sphere"
    ))

    # Add coordinate axes
    axis_length = 1.2
    fig.add_trace(go.Scatter3d(
        x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='gray', width=2),
        showlegend=False, name="X-axis"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
        mode='lines', line=dict(color='gray', width=2),
        showlegend=False, name="Y-axis"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
        mode='lines', line=dict(color='gray', width=2),
        showlegend=False, name="Z-axis"
    ))

    # Add qubit state vectors
    colors = px.colors.qualitative.Set1
    for i, (qubit, values) in enumerate(expectation_values.items()):
        x, y, z = values['x'], values['y'], values['z']
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color=color, width=6),
            marker=dict(size=[2, 8], color=[color, color]),
            name=f'Qubit {qubit}',
            text=[f'Origin', f'Qubit {qubit}<br>x={x:.3f}<br>y={y:.3f}<br>z={z:.3f}']
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-1.3, 1.3], title="X"),
            yaxis=dict(range=[-1.3, 1.3], title="Y"),
            zaxis=dict(range=[-1.3, 1.3], title="Z"),
            aspectmode="cube"
        ),
        width=800,
        height=600
    )
    
    return fig

def plot_entanglement_heatmap(qc: QuantumCircuit, num_qubits: int):
    """
    Create a heatmap showing potential entanglement between qubits
    based on gate connectivity in the circuit.
    """
    # Initialize connectivity matrix
    connectivity = np.zeros((num_qubits, num_qubits))
    
    # Analyze circuit for two-qubit gates
    for instruction in qc.data:
        if len(instruction.qubits) == 2:  # Two-qubit gate
            # q1, q2 = instruction.qubits[0].index, instruction.qubits[1].index
            q1 = qc.find_bit(instruction.qubits[0]).index
            q2 = qc.find_bit(instruction.qubits[1]).index
            connectivity[q1, q2] += 1
            connectivity[q2, q1] += 1
    
    fig = go.Figure(data=go.Heatmap(
        z=connectivity,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Qubit Connectivity Heatmap",
        xaxis_title="Qubit Index",
        yaxis_title="Qubit Index"
    )
    
    return fig

def plot_gate_count_analysis(qc: QuantumCircuit):
    """Analyze and visualize gate usage in the circuit."""
    gate_counts = {}
    
    for instruction in qc.data:
        gate_name = instruction.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    if gate_counts:
        fig = go.Figure(data=[
            go.Bar(x=list(gate_counts.keys()), y=list(gate_counts.values()))
        ])
        fig.update_layout(
            title="Gate Usage Analysis",
            xaxis_title="Gate Type",
            yaxis_title="Count"
        )
        return fig
    else:
        return None

def create_large_random_circuit(num_qubits: int = 30, depth: int = 20):
    """Create a random circuit for testing with many qubits."""
    qc = QuantumCircuit(num_qubits)
    
    np.random.seed(42)  # For reproducibility
    
    for layer in range(depth):
        # Add random single-qubit gates
        for qubit in range(num_qubits):
            if np.random.random() < 0.3:  # 30% chance of single-qubit gate
                gate_choice = np.random.choice(['h', 'x', 'y', 'z', 's', 't'])
                if gate_choice == 'h':
                    qc.h(qubit)
                elif gate_choice == 'x':
                    qc.x(qubit)
                elif gate_choice == 'y':
                    qc.y(qubit)
                elif gate_choice == 'z':
                    qc.z(qubit)
                elif gate_choice == 's':
                    qc.s(qubit)
                elif gate_choice == 't':
                    qc.t(qubit)
        
        # Add random two-qubit gates
        for _ in range(num_qubits // 3):  # Add some entangling gates
            q1, q2 = np.random.choice(num_qubits, 2, replace=False)
            gate_choice = np.random.choice(['cx', 'cz', 'swap'])
            if gate_choice == 'cx':
                qc.cx(q1, q2)
            elif gate_choice == 'cz':
                qc.cz(q1, q2)
            elif gate_choice == 'swap':
                qc.swap(q1, q2)
    
    return qc

# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Qubit Quantum Visualizer", layout="wide")

st.title("🔮 Enhanced Quantum State Visualiz" \
"er (25 Qubits)")
st.write("Advanced visualization techniques for large quantum circuits")

# Sidebar for configuration
st.sidebar.header("Configuration")
visualization_method = st.sidebar.selectbox(
    "Choose visualization method:",
    ["Sampling-based Bloch Vectors", "Stabilizer Analysis", "Circuit Analysis", "All Methods"]
)

max_qubits_display = st.sidebar.slider("Max qubits to display:", 1, 30, 10)
sampling_shots = st.sidebar.slider("Sampling shots:", 1000, 16384, 8192)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Circuit Selection")
    mode = st.radio("Choose mode:", [
        "Predefined Circuits", 
        "Large Random Circuit", 
        "Custom Code", 
        "Build from Gates"
    ])

    if mode == "Predefined Circuits":
        example = st.selectbox("Choose circuit:", [
            "Hadamard", "Bell State", "GHZ", "QFT", "Grover"
        ])
        if example == "Hadamard":
            qc = hadamard_circuit()
        elif example == "Bell State":
            qc = bell_state_circuit()
        elif example == "GHZ":
            qc = ghz_state_circuit()
        elif example == "QFT":
            qc = qft_circuit()
        elif example == "Grover":
            qc = grover_circuit()

    elif mode == "Large Random Circuit":
        num_qubits = st.slider("Number of qubits:", 10, 50, 30)
        circuit_depth = st.slider("Circuit depth:", 5, 50, 20)
        if st.button("Generate Random Circuit"):
            qc = create_large_random_circuit(num_qubits, circuit_depth)
            st.session_state['current_circuit'] = qc
        
        if 'current_circuit' in st.session_state:
            qc = st.session_state['current_circuit']
        else:
            qc = create_large_random_circuit(30, 20)

    elif mode == "Custom Code":
        user_code = st.text_area(
            "Write your circuit here (use variable name 'qc'):",
            """
from qiskit import QuantumCircuit
import numpy as np

# Create a 30-qubit GHZ-like state
qc = QuantumCircuit(30)
qc.h(0)
for i in range(29):
    qc.cx(i, i+1)
            """, 
            height=250
        )

        try:
            local_scope = {'QuantumCircuit': QuantumCircuit, 'np': np}
            exec(user_code, local_scope, local_scope)
            if "qc" in local_scope and isinstance(local_scope["qc"], QuantumCircuit):
                qc = local_scope["qc"]
            else:
                st.error("❌ No valid QuantumCircuit named 'qc' found.")
                qc = create_large_random_circuit(10, 5)
        except Exception as e:
            st.error(f"❌ Error in your code: {e}")
            qc = create_large_random_circuit(10, 5)

    elif mode == "Build from Gates":
        num_qubits = st.number_input("Number of Qubits", min_value=1, max_value=100, value=30)
        qc = QuantumCircuit(num_qubits)

        if "gates_added" not in st.session_state:
            st.session_state.gates_added = []

        col_gate, col_targets = st.columns(2)
        with col_gate:
            selected_gate = st.selectbox("Choose a gate", 
                ["h", "x", "y", "z", "s", "t", "cx", "cz", "swap", "ccx"])
        with col_targets:
            targets = st.text_input("Target qubits (comma separated)", "0")

        if st.button("Add Gate"):
            try:
                target_list = [int(t.strip()) for t in targets.split(",") if t.strip()]
                if any(t >= num_qubits or t < 0 for t in target_list):
                    st.error("❌ Invalid target qubit index.")
                else:
                    st.session_state.gates_added.append((selected_gate, target_list))
            except Exception as e:
                st.error(f"❌ Error adding gate: {e}")

        # Rebuild circuit
        for gate, targets in st.session_state.gates_added:
            try:
                if gate == "h":
                    qc.h(targets[0])
                elif gate == "x":
                    qc.x(targets[0])
                elif gate == "y":
                    qc.y(targets[0])
                elif gate == "z":
                    qc.z(targets[0])
                elif gate == "s":
                    qc.s(targets[0])
                elif gate == "t":
                    qc.t(targets[0])
                elif gate == "cx" and len(targets) >= 2:
                    qc.cx(targets[0], targets[1])
                elif gate == "cz" and len(targets) >= 2:
                    qc.cz(targets[0], targets[1])
                elif gate == "swap" and len(targets) >= 2:
                    qc.swap(targets[0], targets[1])
                elif gate == "ccx" and len(targets) >= 3:
                    qc.ccx(targets[0], targets[1], targets[2])
            except Exception as e:
                st.error(f"Error applying gate {gate}: {e}")

        if st.button("Clear Circuit"):
            st.session_state.gates_added = []

with col2:
    st.subheader("Circuit Information")
    if 'qc' in locals():
        st.write(f"Qubits: {qc.num_qubits}")
        st.write(f"Gates: {qc.size()}")
        st.write(f"Depth: {qc.depth()}")
        
        # Show simplified circuit representation for large circuits
        if qc.num_qubits <= 10:
            st.text("Circuit Diagram:")
            st.text(str(qc.draw(fold=-1)))
        elif qc.num_qubits > 10:
            st.text("Partial Circuit Diagram (first 10 qubits):")
            # st.text(str(qc.draw(fold=-1, output='text', scale=0.5)[:1000]))  # Truncate for display
            st.text(str(qc.draw(fold=-1, output="text", scale=0.5))[:1000])


# Visualization Section
if 'qc' in locals():
    st.subheader("🎯 Quantum State Visualization")
    
    # Select which qubits to analyze
    qubit_indices = list(range(min(max_qubits_display, qc.num_qubits)))
    selected_qubits = st.multiselect(
        "Select qubits to visualize:", 
        range(qc.num_qubits),
        default=qubit_indices
    )
    
    if not selected_qubits:
        st.warning("Please select at least one qubit to visualize.")
    else:
        tab1, tab2, tab3 = st.tabs(["Bloch Spheres", "Circuit Analysis", "Advanced"])
        
        with tab1:
            if visualization_method in ["Sampling-based Bloch Vectors", "All Methods"]:
                with st.spinner("Computing Bloch vectors using sampling..."):
                    expectation_values = get_sampling_based_expectation_values(
                        qc, selected_qubits, sampling_shots
                    )
                    
                    if expectation_values:
                        fig = plot_bloch_sphere_enhanced(
                            expectation_values, 
                            f"Sampling-based Bloch Vectors ({sampling_shots} shots)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show numerical values
                        st.subheader("Expectation Values")
                        for qubit in selected_qubits:
                            if qubit in expectation_values:
                                vals = expectation_values[qubit]
                                st.write(f"Qubit {qubit}: X={vals['x']:.3f}, Y={vals['y']:.3f}, Z={vals['z']:.3f}")
        
        with tab2:
            # Gate analysis
            gate_fig = plot_gate_count_analysis(qc)
            if gate_fig:
                st.plotly_chart(gate_fig, use_container_width=True)
            
            # Connectivity analysis
            if qc.num_qubits <= 50:  # Only for reasonable sizes
                connectivity_fig = plot_entanglement_heatmap(qc, qc.num_qubits)
                st.plotly_chart(connectivity_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Advanced Analysis")
            
            if st.button("Analyze Circuit Complexity"):
                # Circuit complexity metrics
                st.write(f"Two-qubit gate count: {sum(1 for inst in qc.data if len(inst.qubits) == 2)}")
                st.write(f"Single-qubit gate count: {sum(1 for inst in qc.data if len(inst.qubits) == 1)}")
                
                # Try to estimate quantum volume (simplified)
                depth = qc.depth()
                width = qc.num_qubits
                est_qv = min(width, depth) ** 2
                st.write(f"Estimated Quantum Volume: {est_qv}")
            
            if st.button("Try Stabilizer Analysis"):
                stabilizer_result = get_stabilizer_tableau(qc)
                if stabilizer_result:
                    st.success("✅ Circuit is stabilizer-simulable!")
                    st.write("This circuit can be efficiently simulated classically.")
                else:
                    st.info("Circuit may not be stabilizer-simulable or analysis failed.")
    
    # Optimization section
    st.subheader("🔧 Circuit Optimization")
    if st.button("Optimize Circuit"):
        try:
            with st.spinner("Optimizing circuit..."):
                original_size = qc.size()
                original_depth = qc.depth()
                
                optimized_qc = transpile(qc, optimization_level=3)
                
                st.success("✅ Circuit optimized!")
                st.write(f"Original: {original_size} gates, depth {original_depth}")
                st.write(f"Optimized: {optimized_qc.size()} gates, depth {optimized_qc.depth()}")
                
                reduction = (original_size - optimized_qc.size()) / original_size * 100
                st.write(f"Gate reduction: {reduction:.1f}%")
                
        except Exception as e:
            st.error(f"Optimization failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
Scalability Notes:
- Sampling-based approach: Works for any number of qubits, uses measurement statistics
- Stabilizer circuits: Exact simulation possible for certain circuit types  
- Circuit analysis: Always available regardless of qubit count
- Memory usage: Minimal for sampling methods, scales with shots rather than qubit count
""")