from qiskit import QuantumCircuit
import numpy as np

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
    qc.cx(0, 2)
    return qc

def teleportation_circuit():
    qc = QuantumCircuit(3, 2)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc

def qft_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cp(np.pi/2, 1, 0)
    qc.cp(np.pi/4, 2, 0)
    qc.h(1)
    qc.cp(np.pi/2, 2, 1)
    qc.h(2)
    qc.swap(0, 2)
    return qc

def grover_circuit():
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    qc.cz(0, 1)
    qc.h(0)
    qc.x(0)
    qc.h(1)
    qc.cz(0, 1)
    qc.h(1)
    qc.x(0)
    qc.h(0)
    qc.h(1)
    return qc
