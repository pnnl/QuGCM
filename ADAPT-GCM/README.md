# ADAPT-GCM

Code and data for the [ADAPT-GCM paper](https://arxiv.org/abs/2312.07691). We use the certain files from the [original ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src) to construct the Hamiltonian and ansatz pool (and the VQE gradient computation provided in the same file) so that the performance comparison in the paper is straightforward and without doubts.

## Files and folders in this path
* `Paper_Data` folder: The folder contains all data for the [ADAPT-GCM paper](https://arxiv.org/abs/2312.07691). See the README inside for details.
* `ADAPT_GCM_Tutorial.ipynb`: The step-by-step tutorial for ADAPT-GCM and quantum resource estimation with a LiH example.
* `environment.yml`: The files that contains all my packages installed in our Anaconda environment. The essential ones are listed in the next section
* `adapt_gcm.py`: The key files that run ADAPT-GCM.
* `mole.py`: The files records the geometryies of molecules used in the paper. MUST be paired with `pyscf_helper.py` in [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src). The part of `mole.py` that related to `pyscf_helper.py` is based on the [h6_sd.py](https://github.com/mayhallgroup/adapt-vqe/blob/master/examples/h6_sd.py) in the same [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src).
* `Demo_Data` folder: The data folder for `ADAPT_GCM_Tutorial.ipynb`


## Prerequisite

### Required Packages
```text
python=3.10.0
numpy==1.23.0
scipy==1.8.1
h5py==3.7.0
qiskit==0.45.0
qiskit-algorithms==0.2.1
openfermion==1.5.1
openfermionpsi4==0.5
pyscf==2.3.0
```

### Required Files from Original ADAPT-VQE Repository

To have an accurate and dirct comparison, we use the following files from [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src)
* `pyscf_helper.py` to create the Hamiltonian from the geometry
* `operator_pools.py` to generate ansatz pool (as discussed in [ADAPT-GCM paper](https://arxiv.org/abs/2312.07691))


Due to upgrades in `openfermion`, `get_sparse_operator()` is not in `openfermion.linalg`. For `operator_pools.py`, please add the following at the top
```
import openfermion.linalg as transforms 
```

Due to upgrades in `pyscf`, `molden` is not in `pyscf` directly. In line 3 of `pyscf_helper.py`, please remove the import of `molden` and add the following at the top
```
from pyscf.tools import molden
```
If the atomic unit is used (such as $H_4$ and $H_6$ in [ADAPT-GCM paper](https://arxiv.org/abs/2312.07691)), please add the following code in line 229 in `pyscf_helper.py`
```
mol.unit = 'B'  # (B, b, Bohr, bohr, AU, au) or (A, a, Angstrom, angstrom, Ang, ang)
```
