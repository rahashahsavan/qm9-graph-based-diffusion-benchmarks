# GraphARM QM9 Training and Generation Guide

## 🧩 Installation

```bash
cd GraphARM
pip install -r requirements_qm9.txt
```
##Test Data Loading
```python
python test_qm9_loading.py
```

This script:

Loads the QM9 dataset

Removes hydrogens (optional)

Displays dataset statistics

## Train the Model
```python
python train_qm9.py
```

Configuration options:

```
REMOVE_HYDROGEN = True → remove hydrogens (default)

REMOVE_HYDROGEN = False → keep hydrogens
```

3️⃣ Generate Molecules
```bash
python generate_molecules.py
```

