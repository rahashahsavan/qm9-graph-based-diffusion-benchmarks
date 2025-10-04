# Usage Example for Molecular Evaluation Pipeline

## Step-by-Step Guide

### 1. Install Dependencies

```bash
pip install -r requirements_metrics.txt
```

Note: Some dependencies are optional. The script will work with just the core dependencies and skip metrics that require unavailable libraries.

### Minimal Installation (Core Metrics Only)

```bash
pip install numpy pandas scipy scikit-learn rdkit-pypi torch
```

### Full Installation (All Metrics)

```bash
pip install -r requirements_metrics.txt
```

---

## 2. Prepare Your Data Files

### Format: `.smi` or `.txt` files

One SMILES per line (molecules WITHOUT hydrogens for QM9):

**generated.smi**
```
CCO
CCC
NCCN
c1ccccc1
CC(=O)O
CCN
NCCO
```

**reference.smi**
```
CCO
CCC
CCCC
CCCCC
CCCCCC
NCCN
NCC
CCN
NCCO
CCCO
```

---

## 3. Run Evaluation

### Basic Command

```bash
python evaluate_metrics.py --generated generated.smi --reference reference.smi
```

### With Custom Output Name

```bash
python evaluate_metrics.py \
    --generated generated.smi \
    --reference reference.smi \
    --output_prefix my_experiment_results
```

### For Testing (Limited Samples)

```bash
python evaluate_metrics.py \
    --generated generated.smi \
    --reference reference.smi \
    --max_samples 1000
```

---

## 4. Example Output

### Console Output

```
Loading SMILES from: generated.smi
Loaded 10000 SMILES
Loading SMILES from: reference.smi
Loaded 50000 SMILES

Evaluating 10000 generated molecules against 50000 reference molecules

Computing validity...
Computing uniqueness...
Computing novelty...
Computing atom stability...
Computing molecular stability...
Computing FCD...
Computing MMD...
Computing NLL...
Computing NSPDK...

Results saved to metrics_results.json
Results saved to metrics_results.csv

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

### JSON Output (`metrics_results.json`)

```json
{
  "validity": {
    "validity": 0.9512,
    "valid_count": 9512,
    "invalid_count": 488,
    "total_count": 10000
  },
  "uniqueness": {
    "uniqueness": 0.8734,
    "unique_count": 8306,
    "valid_count": 9512,
    "total_count": 10000
  },
  "novelty": {
    "novelty": 0.7156,
    "novel_count": 5945,
    "valid_generated_count": 8306,
    "total_count": 10000
  },
  "atom_stability": {
    "atom_stability": 0.9856,
    "stable_atoms": 125438,
    "total_atoms": 127234
  },
  "mol_stability": {
    "mol_stability": 0.9234,
    "stable_molecules": 8782,
    "total_molecules": 9512
  },
  "fcd": {
    "fcd": 2.3456,
    "valid_generated_count": 9512,
    "valid_reference_count": 50000
  },
  "mmd": {
    "mmd": 0.0234,
    "generated_samples": 9512,
    "reference_samples": 50000
  },
  "nll": {
    "nll": 12.4567,
    "sample_count": 9512
  },
  "nspdk": {
    "nspdk": 0.8923,
    "generated_samples": 9512,
    "reference_samples": 50000
  }
}
```

### CSV Output (`metrics_results.csv`)

```csv
metric,submetric,value
validity,validity,0.9512
validity,valid_count,9512
validity,invalid_count,488
validity,total_count,10000
uniqueness,uniqueness,0.8734
uniqueness,unique_count,8306
novelty,novelty,0.7156
novelty,novel_count,5945
atom_stability,atom_stability,0.9856
mol_stability,mol_stability,0.9234
fcd,fcd,2.3456
mmd,mmd,0.0234
nll,nll,12.4567
nspdk,nspdk,0.8923
```

---

## 5. Analyzing Results

### Loading in Python

```python
import json
import pandas as pd

# Load JSON
with open('metrics_results.json', 'r') as f:
    metrics = json.load(f)

print(f"Validity: {metrics['validity']['validity']:.4f}")
print(f"Uniqueness: {metrics['uniqueness']['uniqueness']:.4f}")
print(f"Novelty: {metrics['novelty']['novelty']:.4f}")

# Load CSV
df = pd.read_csv('metrics_results.csv')
print(df)
```

### Comparing Multiple Experiments

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple experiments
exp1 = pd.read_csv('experiment1_results.csv')
exp2 = pd.read_csv('experiment2_results.csv')

# Compare validity
exp1_validity = exp1[exp1['submetric'] == 'validity']['value'].iloc[0]
exp2_validity = exp2[exp2['submetric'] == 'validity']['value'].iloc[0]

print(f"Experiment 1 Validity: {exp1_validity:.4f}")
print(f"Experiment 2 Validity: {exp2_validity:.4f}")
```

---

## 6. Common Use Cases

### Evaluating DisCo Generated Molecules

```bash
python evaluate_metrics.py \
    --generated DisCo/Generated_Molecules/disco_qm9_smiles.txt \
    --reference qm9_test_noH.smi \
    --output_prefix disco_evaluation
```

### Evaluating GraphARM Generated Molecules

```bash
python evaluate_metrics.py \
    --generated GraphARM/generated_molecules.smi \
    --reference qm9_test_noH.smi \
    --output_prefix grapharm_evaluation
```

### Evaluating MUDiff Generated Molecules

```bash
python evaluate_metrics.py \
    --generated MUDiff/generated_molecules.smi \
    --reference qm9_test_noH.smi \
    --output_prefix mudiff_evaluation
```

---

## 7. Extracting QM9 Reference SMILES

If you need to create a reference file from QM9 dataset:

```python
# Example: Extract SMILES from QM9 (without hydrogens)
from rdkit import Chem

# Load QM9 molecules and remove hydrogens
smiles_list = []
for mol in qm9_molecules:
    mol_no_h = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol_no_h)
    smiles_list.append(smiles)

# Save to file
with open('qm9_reference_noH.smi', 'w') as f:
    for smiles in smiles_list:
        f.write(f"{smiles}\n")
```

---

## 8. Troubleshooting

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

## 9. Expected Metric Ranges

Based on literature for QM9 molecular generation:

- **Validity**: >90% (good), >95% (excellent)
- **Uniqueness**: >80% (good), >90% (excellent)
- **Novelty**: 50-90% (depends on training set size)
- **Atom Stability**: >95% (good)
- **Mol Stability**: >90% (good)
- **FCD**: <5 (excellent), <10 (good), <20 (acceptable)
- **MMD**: <0.05 (excellent), <0.1 (good)
- **NSPDK**: >0.8 (good similarity)

---

## 10. Integrating with Your Workflow

### As Part of Training Pipeline

```bash
#!/bin/bash

# Train model
python train_model.py --config config.yaml

# Generate molecules
python generate_molecules.py --checkpoint model.pt --output generated.smi

# Evaluate
python evaluate_metrics.py \
    --generated generated.smi \
    --reference qm9_reference.smi \
    --output_prefix experiment_$(date +%Y%m%d)

# Archive results
mkdir -p results/experiment_$(date +%Y%m%d)
mv experiment_*.json experiment_*.csv results/experiment_$(date +%Y%m%d)/
```

---

**Happy evaluating! 🎉**

