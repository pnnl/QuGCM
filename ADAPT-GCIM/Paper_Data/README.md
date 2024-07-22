# Data folder for [ADAPT-GCIM paper](https://arxiv.org/abs/2312.07691)


## Data for ADAPT-VQE-GCIM
Only squared H4 is not compressed to serve a convenient illustration and all data for other five molecules are compressed. 

### Inside of folder `IterMats`
In `IterMats`, projected Hamiltonian matrix H and overlap matrix S are recored, as well as time used to compute gradient, VQE, GCIM computation in each iteration. The array of coefficients in matrix exponential are also recorded.

### Outside of folder `IterMats`

General
* `printout.txt` gives all printout in ADAPT-VQE along with ADAPT-VQE-GCIM and ADAPT-VQE-GCIM(1). NOTE: due to the change of the implementation of the custom eigensolver, the eigenvalues from GCIM in this file are not accurate. For the accurate values, please use `GCM_DIFFS_EIGRECON.npy`.
* `optimized_coeffs.npy` stores the optimal parameters from ADAPT-VQE after convergence (also partially used by ADAPT-GCIM, as described in paper)
* `selected_indices.npy` stores the index of selected ansatzes based on VQE gradients, need to pair with spin-adapted pool, i.e., the pool from code `operator_pools.singlet_GSD()` in `ADAPT_GCIM_Demo.ipynb`
* `times_XXX.npy` stores computation time for XXX, including gradeint evaluaiton, VQE optimization, GCIM matrix formation, GCIM eigensolving

ADAPT-VQE-GCIM Part
* `GCM_DIFFS_EIGRECON.npy` stores the errors of estimated ground-state energies in all iterations by ADAPT-VQE-GCIM
* `GCM_OneShot.npy` records the eigenvalue from the final one-shot GCIM (i.e., ADAPT-VQE-GCIM(1) in paper)
* `GCM_OneShotDiff.npy` stores the error of estimated ground-state energy by ADAPT-VQE-GCIM(1)
* `GCM_SIZE.npy` gives the size of bases in all iterations by ADAPT-VQE-GCIM

ADAPT-VQE part
* `VQE_DIFFS.npy` stores the errors of estimated ground-state energies in all iterations by ADAPT-VQE-GCIM
* `VQE_NITERS.npy` stores the number of iterations to converge by ADAPT-VQE (yes, it just store a integer)

Quantum Resources
* `VQE_opcounts.json` stores the number of basis gates needed to construct single basis operator and product basis operator in each iteration (note that the product basis operator is the same as the one used in ADAPT-VQE)
* `VQE_opcounts_opt.json` is similar to `VQE_opcounts.json`, but the numbers are optimized by `Qiskit`
* `VQE_Ansatz_cnots.npy` is the same as `VQE_opcounts.json` but only contains number of CNOTs
* `VQE_Ansatz_cnots_opt.npy` is the same as `VQE_opcounts_opt.json` but only contains number of CNOTs


## Data for ADAPT-GCIM

Most of the files are just ADAPT-GCIM counter parts of those in `ADAPT-VQE-GCIM` folder with a change of prefix from `VQE` to `GCM`. The only new name is
* `GCM_EVS.npy` stores lowest eigenvalues from ADAPT-GCIM in all iterations.