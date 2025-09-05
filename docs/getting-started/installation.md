# Installation

## Requirements

DRN requires Python 3.8 or later and has the following dependencies:

- PyTorch (`torch`)
- PyTorch Lightning (`lightning`) 
- NumPy (`numpy`)
- Pandas (`pandas`)
- Scikit-learn (`scikit-learn`)
- Statsmodels (`statsmodels`)
- SHAP (`shap`)
- Matplotlib (`matplotlib`)
- Seaborn (`seaborn`)
- tqdm (`tqdm`)

## Install from PyPI

The easiest way to install DRN is using pip:

```bash
pip install drn
```

## Install for Development

If you want to contribute to DRN or need the latest development features:

```bash
git clone https://github.com/your-org/drn.git
cd drn
pip install -e .[dev]
```

The `[dev]` option installs additional development dependencies:
- `black` - Code formatting
- `isort` - Import sorting
- `mypy` - Type checking
- `pytest` - Testing framework
- `pre-commit` - Git hooks
- `ipykernel` - Jupyter notebook support
- `jupytext` - Jupyter notebook version control

## Verify Installation

To verify your installation is working correctly:

```python
import drn
from drn import GLM, DRN
import torch

print(f"DRN version: {drn.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Test basic functionality
glm = GLM("gaussian")
print("✓ GLM model created successfully")

drn_model = DRN(glm, cutpoints=[0, 1, 2])
print("✓ DRN model created successfully")
```

## GPU Support

DRN automatically detects and uses GPU acceleration when available. PyTorch with CUDA support is recommended for large datasets:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")
```

## Common Installation Issues

### Issue: Module not found after installation

**Solution**: Make sure you're using the correct Python environment:
```bash
which python
pip list | grep drn
```

### Issue: CUDA compatibility problems

**Solution**: Install the correct PyTorch version for your CUDA version:
```bash
python -c "import torch; print(torch.version.cuda)"
```

### Issue: Memory issues during installation

**Solution**: Increase pip cache or use `--no-cache-dir`:
```bash
pip install --no-cache-dir drn
```

### Issue: Permission denied on macOS/Linux

**Solution**: Use virtual environment or user installation:
```bash
pip install --user drn
```

## Virtual Environment Setup

We recommend using a virtual environment:

```bash
# Using venv
python -m venv drn-env
source drn-env/bin/activate  # On Windows: drn-env\Scripts\activate
pip install drn

# Using conda
conda create -n drn-env python=3.9
conda activate drn-env
pip install drn
```

## Docker Installation

For containerized environments:

```dockerfile
FROM python:3.9-slim

RUN pip install drn

WORKDIR /app
COPY . .

CMD ["python", "your_script.py"]
```

## Next Steps

Once installation is complete, head to the [Quick Start](quickstart.md) guide to learn the basics of using DRN.