# Realigning Quantum Architecture Search with a Top-k Focused Training Paradim

This repository provides the official implementation of the paper:

> **Realigning Quantum Architecture Search with a Top-k Focused Training Paradim**  
> (Submission to Physical Review Applied)

We introduce a new training paradigm for Quantum Architecture Search (QAS) centered on a differentiable, top-heavy ranking loss. This approach directly aligns the model's training with the primary goal of QAS: identifying a small set of elite, top-performing quantum circuits, rather than perfectly ranking all candidates. Our framework is enabled by a novel Circuit-native Transformer model and a self-supervised pre-training strategy, establishing a more effective and purpose-driven method for discovering optimal quantum architectures.

📚 Built upon [TensorCircuit](https://tensorcircuit.readthedocs.io/en/latest/index.html).

---

## 🧩 Key Features

- **Top-tier-focused NDCG Loss** – A differentiable ranking loss that directly optimizes top-k accuracy for elite quantum architectures.

  **Circuit-native Transformer** – Transformer with unitary-flow attention and depth-aware encoding tailored to quantum circuit DAGs.

  **RF-based Self-Supervised Pretraining** – Learns functionally-aware circuit representations by predicting Relative Fluctuation before fine-tuning.

---

## 📦 Installation

**Prerequisites**

- Python 3.9

- Pip

  **Install dependencies**

  ```
  pip install -r requirements.txt
  ```

Setup should now be complete.

---

## 🚀 Example Usage

### 1. Generate Candidate Circuits

Generate candidate quantum circuits for different tasks and construct datasets by computing the corresponding ground-state energies.

Run the following script file。

```bash
sampling.sh
```

```bash
VQETrainer.sh
```

- `sampling.sh` samples quantum circuits.
- `VQETrainer.sh` computes the ground-state energy of quantum circuits.

---

### 2. Compute Relative Fluctuation (RF) Proxy

Compute the RF metric using a large number of unlabeled quantum circuits.

```bash
RF_proxy_calculator.sh
```

Relative Fluctuation (RF)[1] is a training-free metric that predicts the learnability of a Quantum Neural Network (QNN). It compares the fluctuation of a QNN’s training landscape with a standard learnable one, identifying issues such as low expressibility, barren plateaus, bad local minima, and overparameterization. 

---

### 3. Pre-training & Fine-tuning Predictor

Pre-training and fine-tuning predictor models designed for quantum circuit evaluation.

```bash
run_experiments.sh
```

---

### 4. Architecture Search Phase

Perform quantum architecture search using the trained predictor to efficiently explore candidate quantum circuits and identify high-performance architectures.

```bash
QAS_run_experiments.sh
```

---

## 📁 Project Structure

```
.
|   circuit_sampler.py
|   config.py # Configuration information
|   energy_calculator.py # Calculate the ground-state energy of quantum circuits in VQE tasks
|   energy_calculator_TFCluster.py # Calculate the ground-state energy of quantum circuits in VQE tasks
|   main.py # Pre-training and fine-tuning the predictor
|   optimize_circuit.py # Optimize quantum circuit parameters
|   QAS.py # Execute the QAS stage
|   QAS_run_experiments.sh
|   quantum_gates.py
|   README.md # README
|   Relative_fluctuation_calculator_Heisenberg_8.py # Calculate RF metrics for different VQE tasks
|   Relative_fluctuation_calculator_TFCluster_8.py # Calculate RF metrics for different VQE tasks
|   Relative_fluctuation_calculator_TFIM_8.py # Calculate RF metrics for different VQE tasks
|   requirements.txt # Python package dependencies
|   results_process_excel.py # Process training and validation results of the predictor
|   RF_proxy_calculator.sh
|   run_experiments.sh
|   sampling.py # Sample quantum circuits
|   sampling.sh
|   VQETrainer.sh
|   VQETrainer_Heisenberg.py # Compute ground-state energy for different VQE tasks in the QAS stage
|   VQETrainer_TFCluster.py
|   VQETrainer_TFIM.py
|           
+---model
|   |   DAGTransformer.py # Transformer model
|   |   GIN_PQAS.py # GIN model used in ablation experiments
|   |   MLP.py
|   |   
|   \---losses
|           SoftNDCGLoss.py # Differentiable ranking loss function
|           
+---utils # Utility library
```

---

## 📊 Used Variational Quantum Eigensolver (VQE) Tasks

- Heisenberg Model
- The 1D Transverse Field Ising Model (TFIM)
- The 1D transverse-field cluster model (TFCM)

---

## 📜  References

```bibtex
[1] Zhang H K, Zhu C, Wang X. Predicting quantum learnability from landscape fluctuation[J]. arXiv preprint arXiv:2406.11805, 2024.
```



