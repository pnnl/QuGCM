# Data folder for [ADAPT-GCM paper](https://arxiv.org/abs/2312.07691)


## Data for ADAPT-VQE-GCM
Only squared H4 is not compressed to serve a convenient illustration and all data for other five molecules are compressed. 

### Inside of folder `IterMats`
In `IterMats`, projected Hamiltonian matrix H and overlap matrix S are recored. The array of coefficients in matrix exponential are also recorded. There are also files named `IterX_bases.npy`. They are records for bases matrices, but stored in a specifc format for ADAPT-VQE-GCM.

### Outside of folder `IterMats`

General
* `printout.txt` gives all printout in ADAPT-VQE along with ADAPT-VQE-GCM and ADAPT-VQE-GCM(1)
* `optimized_coeffs.npy` stores the optimal parameters from ADAPT-VQE after convergence (also partially used by ADAPT-GCM, as described in paper)
* `selected_indices.npy` stores the index of selected ansatzes based on VQE gradients, need to pair with spin-adapted pool, i.e., the pool from code `operator_pools.singlet_GSD()` in `ADAPT_GCM_Tutorial.ipynb`

ADAPT-VQE-GCM Part
* `GCM_DIFFS.npy` stores the errors of estimated ground-state energies in all iterations by ADAPT-VQE-GCM
* `GCM_OneShot.npy` records the eigenvalue from the final one-shot GCM (i.e., ADAPT-VQE-GCM(1) in paper)
* `GCM_OneShotDiff.npy` stores the error of estimated ground-state energy by ADAPT-VQE-GCM(1)
* `GCM_SIZE.npy` gives the size of bases in all iterations by ADAPT-VQE-GCM

ADAPT-VQE part
* `VQE_DIFFS.npy` stores the errors of estimated ground-state energies in all iterations by ADAPT-VQE-GCM
* `VQE_NITERS.npy` stores the number of iterations to converge by ADAPT-VQE (yes, it just store a integer)

Quantum Resources
* `VQE_opcounts.json` stores the number of basis gates needed to construct single basis operator and product basis operator in each iteration (note that the product basis operator is the same as the one used in ADAPT-VQE)
* `VQE_opcounts_opt.json` is similar to `VQE_opcounts.json`, but the numbers are optimized by `Qiskit`
* `VQE_Ansatz_cnots.npy` is the same as `VQE_opcounts.json` but only contains number of CNOTs
* `VQE_Ansatz_cnots_opt.npy` is the same as `VQE_opcounts_opt.json` but only contains number of CNOTs


## Data for ADAPT-GCM

Most of the files are just ADAPT-GCM counter parts of those in `ADAPT-VQE-GCM` folder with a change of prefix from `VQE` to `GCM`. The only new name is
* `GCM_EVS.npy` stores lowest eigenvalues from ADAPT-GCM in all iterations.