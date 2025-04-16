# Protein Dynamics using GNM
This script computes protein dynamics using the Gaussian Network Model (GNM). It retrieves a PDB structure, extracts Cα coordinates for specified chains, constructs a Kirchhoff matrix with a 7.0 Å cutoff, performs eigen decomposition, and calculates the average Mean Square Fluctuations (MSF) from selected vibrational modes.

## Quick Start
### Dependencies:
Install required packages:

	pip install numpy pandas biopython plotly
### Usage:
- Set PDBname to your PDB ID (e.g., "1L7V").

- Define mode_set (e.g., list(range(1, 11)) for the first 10 modes).

- Specify the chains with chainID (e.g., 'ABCD').
### Run the Script:
Execute the script to download the structure, compute eigenvalues/MSF, and display interactive plots for both eigenvalues and MSF profiles.
