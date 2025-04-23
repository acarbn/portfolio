# StructureComparer
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZrQmv9RXxJhA4DRqn-3jZC2NwmYq3_QM?usp=sharing)

A lightweight Python utility on Colab to download, align, visualize, and compare two protein structures (calculating a distance vector between corresponding CA atoms of two given proteins) from the RCSB PDB. Built on Biopython and py3Dmol, it computes per‐residue RMSDs and plots a residue‐by‐residue difference profile.

---

## Features

- **Automatic PDB retrieval** by PDB ID  
- **Chain‐by‐chain CA‐atom superposition**  
- **Aligned structure export** (PDB format)  
- **Interactive 3D visualization** in Colab via py3Dmol  
- **Residue‐wise distance vector** computation  
- **Distance profile plotting** with Plotly, including chain‐boundary markers  

---

## Dependencies

- Python 3.7+  
- [Biopython](https://biopython.org/)  
- [py3Dmol](https://github.com/3dmol/3Dmol.js)  
- [NumPy](https://numpy.org/)  
- [Plotly](https://plotly.com/python/)  
- (Optional, in Colab) `google.colab` for downloads  

Install with:

```bash
pip install biopython py3Dmol numpy plotly
```
## Usage
```python
# Instantiate with mobile vs. reference PDB IDs and their chains
comparer = StructureComparer(
    PDBmobile="1ake",
    chainmobile="A",
    PDBref="4ake",
    chainref="A"
)

# 1. Download & parse structures
comparer.parse_structures()

# 2. Align mobile onto reference (CA‐atom superposition)
comparer.align_structures()

# 3. (Optional) Save aligned PDB files locally and trigger Colab download
comparer.save_aligned_structures()

# 4. Visualize in‐notebook (returns a py3Dmol view; call .show() in Colab)
view = comparer.visualize()
view.show()

# 5. Compute residue‐wise RMSD vector
diffs = comparer.diff_vector() # `diffs` is a list of (residue_index, distance) tuples

# 6. Plot the difference profile (marks chain boundaries every `window` residues)
comparer.plot_diff_vector(window=25)
```
## Class & Method Reference
### StructureComparer(PDBmobile, chainmobile, PDBref, chainref)
- PDBmobile / PDBref: Four‐character PDB IDs (e.g. "1AKE")
- chainmobile / chainref: String of chain letters (e.g. "ABCD" or "A")

### parse_structures()
Downloads PDB files via Biopython’s PDBList, parses them into Structure objects.

### align_structures()
- Extracts CA atoms from matching residues in each chain.

- Superimposes mobile onto reference using Superimposer.

- Stores aligned PDB contents as in‐memory strings (mov_pdb, ref_pdb).

### visualize() → py3Dmol.view
Returns an 800×600 py3Dmol viewer with:

- Reference model in blue

- Mobile model in red

Use .show() in Colab to render.

### save_aligned_structures()
Saves two files to disk:

- {PDBmobile}_aligned.pdb

- {PDBref}_ref.pdb

In Colab, triggers browser download via google.colab.files.

### diff_vector() → List[(res_id, rmsd)]
Computes per‐residue distance between CA atoms after alignment. Returns a list of tuples where:

- res_id: integer residue number from the reference chain

- rmsd: float distance in Å

### plot_diff_vector(window)
Plots the RMSD vector with Plotly:

- X‐axis: residue number

- Y‐axis: CA‐atom distance (Å)

- Vertical dashed lines at chain boundaries

- Custom tick marks every window residues

## Notes & Tips
- Ensure chain strings match the PDB file’s chain IDs.

- If no CA‐atoms are found (e.g. non‐standard residues), alignment will fail—check your PDB.

- Use larger window values for long proteins to avoid overcrowding tick labels.

## License
MIT License

## Acknowledgements
- Biopython for structure parsing & superposition

- py3Dmol for interactive molecular graphics

- Plotly for flexible, web‐based plotting



