# GNM Mean Square Fluctuations-Sequence Enrichment Score Correlation Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jzN_ZdrjgFe7SL9t9XzMbg7gFy-gwQaj?usp=sharing)

This Google Colab notebook analyzes the correlation between Gaussian Network Model (GNM) mean square fluctuations (MSF) and experimental enrichment scores from FACS.

## 🧬 Features

- Upload enrichment data from Google Drive or your local computer
- Calculate GNM MSF for selected modes using a PDB structure
- Correlate MSF with enrichment scores for each mode
- Visualize and fit regression lines for each GNM mode
- Sort and report slope values to indicate strength of correlation

## 🛠 Requirements

All dependencies are pre-installed in Colab, but the notebook uses:

- `biopython`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- Custom module: `utilsclass.py` (must contain a class named `GNM`)

## 📁 Folder Structure

Make sure your Google Drive has the following structure:

```
MyDrive/
├── enrichment_example.csv      # Example enrichment data (CSV format)
└── portfolio/
    └── utilsclass.py           # Custom module with GNM class
```

## 🚀 Usage

1. Open the notebook in Google Colab.
2. Run all cells.
3. When prompted:
   - Type `c` to upload a file from your local machine.
   - Type `g` to use the `enrichment_example.csv` file from Google Drive.
4. MSF vs Enrichment plots will be generated for selected GNM modes.
5. A summary table of regression slopes will be displayed at the end.

## 📦 Inputs and Parameters
- PDBname: PDB ID of the structure (e.g., "1EU8")

- chainID: Chain identifier (e.g., "A")

- mode_set: List of GNM mode indices (e.g., [1,2,...,10])

- rcut_gnm: GNM cutoff radius in Å (default 10.0)

## 📊 Example Output

Each subplot shows the scatter of MSF vs enrichment scores for a GNM mode. A red regression line indicates correlation strength.

At the end:

```
 mode_no      slopes
       1 -247.841385
       5  -18.580921
       6   -8.501849
      10   -0.920148
       2    3.853023
       8    7.908011
       9   32.268346
       3   36.890167
       4   60.041799
       7   60.806185
```

## 🧪 Notes

- MSF and enrichment must share a common column named `ResidueNo`.

## 📬 Contact

For questions or improvements, feel free to fork or open an issue.

## 📄 License 

This project is licensed under the MIT License.
