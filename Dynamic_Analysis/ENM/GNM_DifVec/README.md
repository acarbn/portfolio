# Finding Functional GNM Modes of Motion 

This Colab notebook performs a comparative analysis between two protein structures (apo and holo forms) using a distance vector approach per CA positions and the Gaussian Network Model (GNM). It computes the per-residue distance vector, aligns the structures, calculates mean-square fluctuations (MSF) for selected GNM modes, and visualizes the results along with cosine similarity metrics between the distance vector and GNM modes.

## 🚀 Features

- **Structure Parsing & Alignment**  
  Uses `Bio.PDB` to download PDB files, parse chains, and superimpose apo and holo conformations.

- **Difference Vector Calculation**  
  Computes the per-residue displacement (difference vector) between aligned structures.

- **GNM Mode Analysis**  
  Leverages a custom `GNM` class to compute eigenvalues, eigenvectors, and per-residue MSFs for specified modes.

- **Visualization with Plotly**  
  Generates subplots of normalized difference vector vs. MSF for each mode, annotated with cosine similarity scores.

- **Summary Table**  
  Constructs a Pandas DataFrame comparing cosine similarities across modes.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Setup & Installation](#setup--installation)  
3. [Notebook Walkthrough](#notebook-walkthrough)  
4. [Configuration](#configuration)  
5. [Results & Interpretation](#results--interpretation)  
6. [Extending the Analysis](#extending-the-analysis)  
7. [License](#license)

---

## Prerequisites

- Python 3.7+  
- Google Colab environment  
- Google Drive account (for mounting and accessing `utilsclass.py`)  
- Internet connection (to fetch PDB files)

## Setup & Installation

1. **Clone this repository**  
   ```bash
   git clone <your-repo-url>.git
   cd <repo-folder>
   ```

2. **Open the Colab notebook**  
   Go to [Google Colab](https://colab.research.google.com/) and upload `Protein_GNM_Analysis.ipynb` or open it directly from your GitHub.

3. **Mount Google Drive**  
   The notebook will prompt you to authorize and mount your Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Install Python dependencies**  
   All required libraries are available in Colab by default. If running locally, install:
   ```bash
   pip install biopython plotly pandas numpy
   ```

## Notebook Walkthrough

1. **Imports & Utilities**  
   ```python
   from Bio.PDB import *
   from google.colab import drive
   import sys
   import numpy as np
   import pandas as pd
   import plotly.graph_objs as go
   from plotly.subplots import make_subplots

   # Mount and add utils path
   drive.mount('/content/drive')
   sys.path.append('/content/drive/MyDrive/portfolio')
   from utilsclass import StructureComparer, GNM
   ```

2. **Define Parameters**  
   ```python
   apoPDB, apoChain   = "1AKE", "A"
   holoPDB, holoChain = "4AKE", "A"
   mode_set           = list(range(1, 11))
   rcut_gnm           = 10  # Ångström
   ```

3. **Structure Parsing & Alignment**  
   ```python
   comparer = StructureComparer(
     PDBmobile=apoPDB, chainmobile=apoChain,
     PDBref=holoPDB,   chainref= holoChain
   )
   comparer.parse_structures()
   comparer.align_structures()
   ```

4. **Compute Difference Vector**  
   ```python
   difvec = comparer.diff_vector()
   indices, distances = zip(*difvec)
   ```

5. **Mode-by-Mode GNM Analysis & Plotting**  
   - Instantiate `GNM`  
   - Compute eigenvalues/eigenvectors  
   - Calculate MSF per residue  
   - Compute cosine similarity to the normalized difference vector  
   - Plot subplots of normalized difference vs. MSF with annotations

6. **Cosine Similarity Summary**  
   ```python
   df = pd.DataFrame({
     'Mode Number': mode_set,
     'Cosine Similarity': cos_sims
   })
   display(df)
   ```

## Configuration

Feel free to adjust:

- **PDB IDs & Chains** (`apoPDB`, `holoPDB`, `apoChain`, `holoChain`)  
- **Mode Set** (`mode_set = [1, 2, …]`)  
- **Cutoff Distance** for GNM (`rcut_gnm`)  
- **Plot Window** (`window` for tick spacing)

## Results & Interpretation

- **Subplots**: Each panel shows the normalized difference vector (red) overlayed with the MSF profile (blue) for a given mode.  
- **Cosine Similarity**: Annotated per subplot, indicating how well each GNM mode’s fluctuation profile aligns with the observed structural displacement.

High positive similarity suggests that mode contributes substantially to the conformational change.

## Extending the Analysis

- **Additional Modes**: Increase `mode_set` to probe more collective motions.  
- **Alternative Models**: Swap GNM for ANM or PCA-based approaches.  
- **Custom Structures**: Apply the same pipeline to other PDB pairs by changing IDs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
