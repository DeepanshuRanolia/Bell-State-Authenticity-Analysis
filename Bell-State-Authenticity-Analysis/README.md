# Comparative Analysis of Different Measures of Authenticity for Bell States

**Bell Violation (CHSH) | Quantum State Tomography (QST) | Visibility**

## Overview

This project investigates and compares three approaches for characterising the authenticity of entangled two-qubit Bell states:

1. **Bell Inequality Violation (CHSH):** Tests whether quantum correlations violate the bound predicted by local hidden-variable theories.
2. **Quantum State Tomography (QST):** Reconstructs the density matrix of a quantum state and evaluates its fidelity with an ideal Bell state.
3. **Visibility:** Measures the contrast of an interference or coincidence fringe and provides an indication of coherence.

The project studies how these measures respond to noise and how effectively they distinguish ideal and imperfect Bell states.

## Bell States

The four maximally entangled two-qubit Bell states are

\[
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt{2}}
\]

\[
|\Phi^-\rangle=\frac{|00\rangle-|11\rangle}{\sqrt{2}}
\]

\[
|\Psi^+\rangle=\frac{|01\rangle+|10\rangle}{\sqrt{2}}
\]

\[
|\Psi^-\rangle=\frac{|01\rangle-|10\rangle}{\sqrt{2}}
\]

## Project Objectives

- Prepare and analyse all four Bell states.
- Calculate Bell-CHSH correlations and investigate Bell inequality violation.
- Reconstruct density matrices using quantum state tomography.
- Calculate fidelity and other state properties.
- Study interference visibility.
- Investigate the effect of depolarising noise on the three measures.
- Compare the strengths and limitations of the different authenticity measures.

## Methodology

The project follows this general workflow:

1. Prepare the four Bell states.
2. Obtain their statevectors and density matrices.
3. Apply a depolarising noise model.
4. Calculate CHSH values, QST fidelity and visibility.
5. Sweep the noise probability and compare the resulting measures.
6. Analyse the results using numerical plots.

For the Werner-state noise model,

\[
\rho(p)=p|\Phi^+\rangle\langle\Phi^+|+(1-p)\frac{I_4}{4}
\]

where \(p\) controls the contribution of the ideal Bell state.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Qiskit
- SciPy
- Jupyter Notebook

## Project Structure

The project is organised into modules for Bell-state preparation, CHSH analysis, quantum state tomography, visibility analysis, noise modelling, comparison and results.

```text
Bell-State-Authenticity-Analysis/
│
├── src/
│   ├── bell_states/
│   ├── bell_violation/
│   ├── tomography/
│   ├── visibility/
│   ├── noise_models/
│   └── comparison/
│
├── notebooks/
├── data/
├── results/
│   └── plots/
│
├── docs/
├── tests/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above is a suggested organisation; adjust it to match the files actually included in the repository.*

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Bell-State-Authenticity-Analysis.git
cd Bell-State-Authenticity-Analysis
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Expected Outcome

The project aims to demonstrate that visibility, CHSH violation and quantum state tomography provide complementary information about entangled-state quality. Visibility characterises fringe contrast, CHSH tests non-local correlations, and QST provides a reconstruction of the complete quantum state.

## References

- J. S. Bell, *On the Einstein Podolsky Rosen Paradox*, Physics Physique Fizika, 1, 195 (1964).
- J. F. Clauser, M. A. Horne, A. Shimony and R. A. Holt, *Proposed Experiment to Test Local Hidden-Variable Theories*, Physical Review Letters, 23, 880 (1969).
- R. F. Werner, *Quantum states with Einstein-Podolsky-Rosen correlations admitting a hidden-variable model*, Physical Review A, 40, 4277 (1989).
- D. F. V. James et al., *Measurement of qubits*, Physical Review A, 64, 052312 (2001).

---
Quantum Cryptography Course Project Simulation
