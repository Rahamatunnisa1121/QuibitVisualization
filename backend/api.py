from flask import Flask, request, jsonify
from qiskit import QuantumCircuit
from backend.bloch_vectors import get_single_qubit_bloch_vectors

app = Flask(__name__)

@app.route("/bloch", methods=["POST"])
def bloch():
    data = request.json
    # Expect serialized circuit in some form (not detailed here)
    qc = QuantumCircuit.from_dict(data["circuit"])
    vectors = get_single_qubit_bloch_vectors(qc)
    return jsonify(vectors)

if __name__ == "__main__":
    app.run(debug=True)
