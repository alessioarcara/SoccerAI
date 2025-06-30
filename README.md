## Installation

<details>
<summary>Click to expand</summary>

Before running the code, you need to install PyTorch and its dependencies. You can choose either the GPU or CPU build depending on your setup. The code has been tested with:

* **PyTorch 2.7.1**
* **CUDA 12.8**
* Optional PyTorch Geometric libraries

### 1. Install PyTorch

| Build               | Command                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| **GPU (CUDA 12.8)** | `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`         |
| **CPU-only**        | `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu`                 |

*Note: Be aware of potential mismatches between CUDA versions when installing.*

---
### 2. PyTorch Geometric stack

Install PyTorch Geometric companion wheels **after** PyTorch:

| Build               | Command                                                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GPU (CUDA 12.8)** | `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu128.html`               |
| **CPU-only**        | `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cpu.html`                 |

---

### 3. Install project dependencies

```bash
pip install .
```

That’s it—you’re ready to run the code!

</details>