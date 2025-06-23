# TorchQPT

**TorchQPT** (PyTorch Quantum Process Tomography) is a comprehensive Python library for quantum information processing and quantum process tomography, built on PyTorch for efficient tensor operations and GPU acceleration. It provides tools for quantum circuit simulation, modeling noisy quantum systems, performing quantum state and process tomography, and working with advanced tensor network structures including Matrix Product States (MPS) and Locally Purified Density Operators (LPDO).

This library is designed for researchers and developers working in quantum computing, quantum information, and quantum process tomography, offering a flexible and intuitive platform for both educational and research purposes.

## 🚀 Core Features

### **Quantum State Representation**
- **`QuantumStateVector`**: Represents pure quantum states using 1D PyTorch tensors
- **`DensityMatrix`**: Represents mixed or pure quantum states using 2D PyTorch tensors
- **Device Support**: Full support for CPU and CUDA devices with automatic device management

### **Quantum Circuit Simulation**
- **`QuantumCircuit`**: Flexible circuit construction with sequential addition of gates and noise channels
- **`CircuitSimulator`**: High-performance circuit simulation supporting both state vector and density matrix evolution
- **Automatic Conversion**: Seamlessly converts state vectors to density matrices when non-unitary operations are applied
- **Multi-qubit Support**: Operations on specified qubits within larger quantum registers

### **Noise Modeling & Quantum Channels**
- **Comprehensive Noise Channels**:
  - Depolarizing Channel (`depolarizing_channel`)
  - Amplitude Damping Channel (`amplitude_damping_channel`)
  - Phase Damping Channel (`phase_damping_channel`)
  - Pauli Channel (`pauli_channel`)
- **Kraus Operator Representation**: All noise channels represented by Kraus operators for accurate quantum dynamics

### **Quantum Process Tomography (QPT)**
- **LPDO Models**: Locally Purified Density Operators for efficient quantum process representation
- **`QPTTrainer`**: Specialized training framework for quantum process tomography
- **Process Fidelity**: Built-in calculation of process fidelity between learned and true channels
- **Noisy Channel Reconstruction**: Advanced capabilities for reconstructing noisy quantum channels

### **Quantum State Tomography (QST)**
- **Linear Inversion Methods**:
  - Single-qubit QST (`qst_linear_inversion_single_qubit`)
  - Multi-qubit QST (`qst_linear_inversion_multi_qubit`)
- **Measurement Framework**: Comprehensive tools for generating measurement projectors and simulating outcomes
- **Basis Support**: X, Y, Z basis measurements and Pauli observable expectation values

### **Tensor Network Methods**
- **Matrix Product States (MPS)**:
  - `MPS` class for one-dimensional tensor networks
  - Product state construction (`MPS.product_state()`)
  - Norm calculation and property inspection
  - Support for arbitrary bond dimensions
- **Advanced Operations**: Bond dimension management, center site tracking, and efficient contractions

### **Data Generation & Training**
- **Synthetic Data Generation**: Tools for generating training data from known quantum channels
- **Measurement Operators**: Comprehensive library of measurement operators and POVM elements
- **Training Utilities**: Batch processing, validation, and early stopping for QPT training

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+ (CPU or GPU)
- NumPy
- pytest (for running tests)

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd TorchQPT

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torchqpt; print('TorchQPT installed successfully!')"
```

## 🎯 Quick Examples

### Basic Quantum Circuit Simulation
```python
from torchqpt.states import QuantumStateVector
from torchqpt.gates import H, CNOT
from torchqpt.circuits import QuantumCircuit
from torchqpt.simulation import CircuitSimulator

# Create a Bell state
psi = QuantumStateVector(2)  # |00⟩
circuit = QuantumCircuit(2)
circuit.add_gate(H(), 0)
circuit.add_gate(CNOT(), (0, 1))

simulator = CircuitSimulator()
bell_state = simulator.run(circuit, psi)
print(f"Bell state: {bell_state.state_vector}")
```

### Quantum Process Tomography
```python
from torchqpt.models import LPDO
from torchqpt.training import QPTTrainer
from torchqpt.data import generate_training_data

# Initialize LPDO model
model = LPDO.random_initialization(
    num_sites=1, physical_dim=2, bond_dim=2, kraus_dim=2
)

# Generate training data
train_data, val_data = generate_training_data(num_qubits=1, num_samples=1000)

# Train the model
trainer = QPTTrainer(model, learning_rate=0.01)
trainer.train(train_data, val_data, num_epochs=100)
```

### Noisy Channel Simulation
```python
from torchqpt.noise import depolarizing_channel
from torchqpt.circuits import QuantumCircuit

# Create circuit with noise
circuit = QuantumCircuit(1)
circuit.add_gate(H(), 0)
circuit.add_kraus(depolarizing_channel(0.1), 0)  # 10% depolarizing noise

# Simulate noisy evolution
simulator = CircuitSimulator()
final_state = simulator.run(circuit, initial_state)
```

## 📁 Project Structure

```
TorchQPT/
├── src/torchqpt/           # Core library code
│   ├── states/             # Quantum state representations
│   ├── circuits/           # Quantum circuit definitions
│   ├── gates.py            # Quantum gate matrices
│   ├── simulation.py       # Circuit simulator
│   ├── noise.py            # Noise channel implementations
│   ├── models/             # Tensor network models (LPDO, MPS)
│   ├── tomography.py       # QST and QPT functions
│   ├── data.py             # Data generation utilities
│   └── training.py         # Training framework
├── examples/               # Example scripts
│   ├── 01_basic_simulation_qsv.py
│   ├── 02_noisy_simulation_dm.py
│   ├── 07_qpt_example.py
│   ├── 08_noisy_channel_qpt.py
│   └── ...
├── tests/                  # Comprehensive test suite
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🧪 Running Examples

The library includes comprehensive examples demonstrating various functionalities:

```bash
# Basic quantum simulation
python examples/01_basic_simulation_qsv.py

# Noisy quantum simulation
python examples/02_noisy_simulation_dm.py

# Quantum process tomography
python examples/07_qpt_example.py

# Noisy channel reconstruction
python examples/08_noisy_channel_qpt.py

# Matrix Product States
python examples/04_mps_basics.py
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_simulation.py
python -m pytest tests/test_tensor_network.py
python -m pytest tests/test_noise.py
```

## 🔬 Recent Improvements

### Version Updates
- **Project Rename**: Updated from legacy name to **TorchQPT**
- **Enhanced LPDO Models**: Improved tensor network contractions and probability calculations
- **Robust Training Framework**: Fixed data format handling and added safety checks
- **Matrix Operations**: Compatible with older PyTorch versions using eigenvalue decomposition
- **Comprehensive Testing**: All test suites now passing with improved error handling

### New Features
- **Quantum Process Tomography**: Complete QPT pipeline with LPDO models
- **Advanced Noise Modeling**: Comprehensive noise channel library
- **Tensor Network Support**: Full MPS implementation with product states
- **Data Generation**: Synthetic data generation for training and validation
- **Process Fidelity**: Built-in fidelity calculations for channel reconstruction

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This library is inspired by quantum simulation frameworks and tensor network methods, aiming to provide an accessible platform for quantum information research and education.

---

**Happy quantum computing with TorchQPT! 🚀**
