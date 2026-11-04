### 1. Environment Setup

#### Requirements
- Python 3.10+
- CUDA 12.4
- GPU with ≥ 8 GB VRAM (16 GB+ recommended for training)

---

#### 1-1. Clone the repository

```bash
git clone https://github.com/prismdevkit/PRISM.git
cd PRISM
```

---

#### 1-2. Create a virtual environment

**Option A — conda (recommended)**
```bash
conda create -n prism python=3.10 -y
conda activate prism
```

**Option B — venv**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

#### 1-3. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies installed:

| Package | Version |
|---------|---------|
| `torch` | 2.4.0+cu124 |
| `torchvision` | 0.19.0+cu124 |
| `pytorch-lightning` | 1.9.5 |
| `open3d` | latest |
| `opencv-python` | latest |
| `wandb` | latest |

> **Note**: The `requirements.txt` pulls PyTorch from `https://download.pytorch.org/whl/cu124`. If you are using a different CUDA version, replace the index URL and torch/torchvision versions accordingly.

---

#### 1-4. Set environment variables

Add the following to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`), or export them before each session:

```bash
export PRISM_DATASET_ROOT=/path/to/PRISM-Dataset   # root directory of the downloaded dataset
export PRISM_DEVKIT_ROOT=/path/to/PRISM            # repo root
```