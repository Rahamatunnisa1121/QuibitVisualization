import streamlit as st
from backend.circuits import hadamard_circuit, bell_circuit, ghz_circuit
from backend.bloch_vectors import get_single_qubit_bloch_vectors
from frontend.bloch_plot import plot_bloch_sphere

st.title("Quantum State Visualizer - Modular UI")

example = st.selectbox("Choose circuit:", ["Hadamard", "Bell", "GHZ"])
if example == "Hadamard":
    qc = hadamard_circuit()
elif example == "Bell":
    qc = bell_circuit()
else:
    qc = ghz_circuit()

st.text(qc.draw())
bloch_vectors = get_single_qubit_bloch_vectors(qc)

for vec in bloch_vectors:
    st.plotly_chart(plot_bloch_sphere(vec, vec["qubit"]))
