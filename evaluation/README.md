# Evaluation Directory

This directory contains the molecular evaluation pipeline for QM9 dataset.

## 📁 Files in this Directory

### Main Scripts
- `evaluate_metrics.py` - Basic evaluation script (SMILES only)
- `evaluate_metrics_enhanced.py` - Enhanced evaluation script (Graph + SMILES)
- `requirements_metrics.txt` - Required Python packages

### Reference Data
- `qm9_reference.smi` - 9,482 QM9 SMILES (without hydrogens)
- `qm9_reference_graphs.pkl` - QM9 graph data (12.5 MB)

### Example Files
- `example_generated.smi` - Sample generated molecules
- `example_reference.smi` - Sample reference molecules

### Documentation
- `README_EVALUATION.md` - Detailed evaluation documentation
- `USAGE_EXAMPLE.md` - Usage examples and tutorials

---

## 🚀 Quick Usage

### 1. Install Dependencies

```bash
pip install -r requirements_metrics.txt
```

### 2. Basic Evaluation (SMILES only)

```bash
python evaluate_metrics.py \
    --generated your_generated.smi \
    --reference qm9_reference.smi \
    --output_prefix results
```

### 3. Enhanced Evaluation (Graph + SMILES)

```bash
python evaluate_metrics_enhanced.py \
    --generated_graphs your_generated_graphs.pkl \
    --generated_smiles your_generated.smi \
    --reference_smiles qm9_reference.smi \
    --reference_graphs qm9_reference_graphs.pkl \
    --output_prefix enhanced_results
```

### 4. Test with Example Data

```bash
# Test basic evaluation
python evaluate_metrics.py \
    --generated example_generated.smi \
    --reference example_reference.smi \
    --output_prefix test_simple

# Test enhanced evaluation
python evaluate_metrics_enhanced.py \
    --generated_smiles example_generated.smi \
    --reference_smiles example_reference.smi \
    --output_prefix test_enhanced
```

---

## 📊 Output Files

Each evaluation run creates:
- `{output_prefix}.json` - Detailed results in JSON format
- `{output_prefix}.csv` - Results in CSV format for analysis

---

## 📈 Metrics Computed

### Basic Evaluation (9 metrics)
1. **Validity** - Fraction of chemically valid molecules
2. **Uniqueness** - Fraction of unique molecules among valid ones
3. **Novelty** - Fraction of molecules not in training set
4. **FCD** - Fréchet ChemNet Distance (distribution similarity)
5. **Atom Stability** - Fraction of atoms with valid configurations
6. **Mol Stability** - Fraction of stable molecules
7. **MMD** - Maximum Mean Discrepancy (distribution distance)
8. **NLL** - Negative Log-Likelihood (likelihood estimation)
9. **NSPDK** - Neighborhood Subgraph Pairwise Distance Kernel (graph similarity)

### Enhanced Evaluation (Additional metrics)
- **Graph Validity** - Structural validity of graphs
- **Graph Topology** - Graph structure analysis
- **Enhanced NSPDK** - Advanced graph kernel similarity

---

## 🔧 Input File Formats

### SMILES Files (.smi or .txt)
One SMILES string per line:
```
CCO
CCC
NCCN
c1ccccc1
CC(=O)O
```

### Graph Files (.pkl)
Pickle files containing graph tuples (X, E):
```python
import pickle

# Each graph is a tuple (X, E)
# X: Node features (atom types) - shape: [n_nodes, n_node_features]
# E: Edge features (bond types) - shape: [n_nodes, n_nodes, n_edge_features]

graphs = [(X1, E1), (X2, E2), ...]

# Save
with open('generated_graphs.pkl', 'wb') as f:
    pickle.dump(graphs, f)
```

---

## 💡 Important Notes

1. **File Input**: Ensure your input files are in the correct format
2. **SMILES**: One SMILES per line, no explicit hydrogens for QM9
3. **Graphs**: Pickle format with tuple (X, E) structure
4. **Reference**: Use `qm9_reference.smi` for comparison

---

## 🎯 Expected Results

### Typical QM9 Test Set
- **Size**: ~13,000 molecules
- **Validity**: >99% (should be nearly all valid)
- **Atom Types**: C, N, O, F (no hydrogens)
- **Size Range**: 3-29 atoms per molecule

### Sample Output
```
============================================================
MOLECULAR GENERATION EVALUATION RESULTS
============================================================

📊 CORE METRICS:
------------------------------
Validity:           0.9512 (9512/10000)
Uniqueness:         0.8734 (8306/9512)
Novelty:            0.7156 (5945/8306)

🔬 QUALITY METRICS:
------------------------------
Atom Stability:     0.9856 (125438/127234)
Mol Stability:      0.9234 (8782/9512)

📈 DISTRIBUTION METRICS:
------------------------------
FCD:                2.3456
MMD:                0.0234
NLL:                12.4567
NSPDK:              0.8923
============================================================
```

---

## 🔧 Troubleshooting

### Issue: "FCD metric will be skipped"
**Solution**: Install fcd-torch
```bash
pip install fcd-torch
```

### Issue: "MOSES not available"
**Solution**: Install moses
```bash
pip install moses
```

### Issue: Memory error with large datasets
**Solution**: Use --max_samples flag
```bash
python evaluate_metrics.py --generated large.smi --reference large_ref.smi --max_samples 10000
```

### Issue: Invalid SMILES in input
The script handles this gracefully and reports counts in the output. Check the validity metrics to see how many molecules were invalid.

---

## 📚 Documentation

- **Main README**: `README_EVALUATION.md` - Comprehensive documentation
- **Usage Examples**: `USAGE_EXAMPLE.md` - Step-by-step tutorials
- **Code**: `evaluate_metrics.py` and `evaluate_metrics_enhanced.py` (well-commented)

For questions about specific metrics, refer to the scientific literature cited in the documentation.

---

## ✅ Checklist for Use

- [ ] Install dependencies: `pip install -r requirements_metrics.txt`
- [ ] Prepare generated.smi file (one SMILES per line, no H)
- [ ] Prepare reference.smi file (QM9 molecules, no H)
- [ ] Test with example files
- [ ] Run evaluation on your data
- [ ] Analyze results

---

**Ready for evaluation! 🚀**
