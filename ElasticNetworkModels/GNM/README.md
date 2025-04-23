# Protein Dynamics Analysis Using GNM

A Python package for Gaussian Network Model (GNM) analysis of protein structures, with eigenmode calculation, mean‑square fluctuation (MSF) profiling, and mode collectivity metrics.

## Features

- **Kirchhoff matrix** construction from PDB structures
- **Eigen-decomposition** for normal modes (eigenvalues & eigenvectors)
- **Mean‑Square Fluctuations (MSF)** calculation per residue
- **Plotting**: eigenvalue spectra & MSF profiles via Matplotlib
- **Collectivity indices** to quantify mode delocalization


## Prerequisites

- Python 3.7+
- [Biopython](https://biopython.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- Plotly

## Usage

Run eigen_collectivity_msf.ipynb in Colab.

## Example Input

- PDBname = "1L7V"
- chainID = 'ABCD'
- mode_set = list(range(1, 11))
- rcut_gnm = 10

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

