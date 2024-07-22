# ADAPT-GCIM

Code and data for the [ADAPT-GCIM paper](https://arxiv.org/abs/2312.07691). We use the certain files from the [original ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src) to construct the Hamiltonian and ansatz pool (and the VQE gradient computation provided in the same file) so that the performance comparison in the paper is straightforward and without doubts. We highly appreciate the accessibility of the source code of ADAPT-VQE. 

## Files and folders in this path
* `Paper_Data` folder: The folder contains all data for the [ADAPT-GCIM paper](https://arxiv.org/abs/2312.07691). See the README inside for details.
* `ADAPT_GCIM_Tutorial.ipynb`: The step-by-step tutorial for ADAPT-GCIM and quantum resource estimation with a LiH example.
* `environment.yml`: The files that contains all my packages installed in our Anaconda environment. The essential ones are listed in the next section
* `adapt_gcim.py`: The key files that run ADAPT-GCIM.
* `mole.py`: The files records the geometryies of molecules used in the paper. MUST be paired with `pyscf_helper.py` in [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src). The part of `mole.py` that related to `pyscf_helper.py` is based on the [h6_sd.py](https://github.com/mayhallgroup/adapt-vqe/blob/master/examples/h6_sd.py) in the same [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src).
* `Demo_Data` folder: The data folder for `ADAPT_GCIM_Tutorial.ipynb`


## Prerequisite

Run the following commands to create an environment.
```
conda env create -f environment.yml
conda activate gcimtest
```

Or, more specially, you can use Python 3.10 or 3.11, and install the following packages.

```text
qiskit==0.45.0
qiskit-algorithms==0.2.1
openfermion==1.5.1
openfermionpsi4==0.5
pyscf==2.6.2
```
Note that, if your computer use Apple M series chip, please install PySCF as
```
ARCHFLAGS="-arch arm64e" pip install pyscf==2.6.2
```

### Required Files from Original ADAPT-VQE Repository

To have an accurate and dirct comparison, we use the following files from [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src)
* `pyscf_helper.py` to create the Hamiltonian from the geometry
* `operator_pools.py` to generate ansatz pool (as discussed in [ADAPT-GCIM paper](https://arxiv.org/abs/2312.07691))


Due to upgrades in `openfermion`, `get_sparse_operator()` is not in `openfermion.transforms`. For `operator_pools.py`, at line 48, please replace `transforms.get_sparse_operator` to 
```
linalg.get_sparse_operator
```

Due to upgrades in `pyscf`, `molden` is not in `pyscf` directly. In line 3 of `pyscf_helper.py`, please remove the import of `molden` and add the following at the top
```
from pyscf.tools import molden
```
If the atomic unit is used (such as $H_4$ and $H_6$ in [ADAPT-GCIM paper](https://arxiv.org/abs/2312.07691)), please add the following code in line 229 in `pyscf_helper.py`
```
mol.unit = 'B'  # (B, b, Bohr, bohr, AU, au) or (A, a, Angstrom, angstrom, Ang, ang)
```

We unfortunately cannot provide
