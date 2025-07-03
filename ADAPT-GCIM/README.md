# ADAPT-GCIM

Code and data for the ADAPT-GCIM paper[(arXiv)](https://arxiv.org/abs/2312.07691) [(published)](https://doi.org/10.1038/s41534-024-00916-8). So it does not include ADAPT-VQE-GCM. . We use the certain files from the [original ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src) to construct the Hamiltonian and ansatz pool (and the VQE gradient computation provided in the same file) so that the performance comparison in the paper is straightforward and without doubts. We highly appreciate the accessibility of the source code of ADAPT-VQE. 

## Files and folders in this path
* `Paper_Data` folder: The folder contains all data for the ADAPT-GCIM paper. See the README inside for details.
* `ADAPT_GCIM_Demo.ipynb`: The step-by-step demo for ADAPT-GCIM and quantum resource estimation with a LiH example.
* `environment.yml`: The files that contains all my packages installed in our Anaconda environment. The essential ones are listed in the next section
* `adapt_gcim.py`: The key files that run ADAPT-GCIM.
* `mole.py`: The files records the geometryies of molecules used in the paper. MUST be paired with `pyscf_helper.py` in [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src). The part of `mole.py` that related to `pyscf_helper.py` is based on the [h6_sd.py](https://github.com/mayhallgroup/adapt-vqe/blob/master/examples/h6_sd.py) in the same [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src).
* `Demo_Data` folder: The data folder for `ADAPT_GCIM_Demo.ipynb`


## Prerequisite

Run the following commands to create an environment.
```
conda env create -f environment.yml
conda activate gcimtest
```

Or, more specifically, you can use Python 3.10 to 3.12, and install the following packages.

```text
qiskit==0.45.0
qiskit-algorithms==0.2.1
openfermion==1.5.1
openfermionpsi4==0.5
pyscf==2.6.2
```
Note that, if your computer uses Apple M series chip, please install PySCF as
```
ARCHFLAGS="-arch arm64e" pip install pyscf==2.6.2
```

### Required Files from my Forked ADAPT-VQE Repository

To have an accurate and dirct comparison, we utilize the following files from [ADAPT-VQE repository](https://github.com/mayhallgroup/adapt-vqe/tree/master/src)
* [`pyscf_helper.py`](https://github.com/Firepanda415/adapt-vqe-for-gcim/blob/master/src/pyscf_helper.py) to create the Hamiltonian from the geometry
* [`operator_pools.py`](https://github.com/Firepanda415/adapt-vqe-for-gcim/blob/master/src/operator_pools.py) to generate ansatz pool (as discussed in ADAPT-GCIM paper)

To run the program under a newer version of PySCF, please use my modified versions for both .py files from [my forked ADAPT-VQE repository](https://github.com/Firepanda415/adapt-vqe-for-gcim/tree/master). You can download them by click the links above.


If the atomic unit is used (such as $H_4$ and $H_6$ in ADAPT-GCIM paper), please add the following code in line 229 in `pyscf_helper.py`
```
mol.unit = 'B'  # (B, b, Bohr, bohr, AU, au) or (A, a, Angstrom, angstrom, Ang, ang)
```
