# Code and Data for QuGCM and ADAPT-GCIM, and their Applications

## Folder "QuGCM" for the original Quantum Generator Coordinated Method (QuGCM)

Code and data for QuGCM paper[(arXiv)](https://arxiv.org/abs/2212.09205) [(published)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.023200)

It is better to directly go to the folder "ADAPT-CGM," unless the interest is on the orginal data of the paper.

## Folder "ADAPT-GCIM" for ADAPT-GCIM

Code for reproducing experiments and data for the ADAPT-GCIM part of the ADAPT-GCIM paper[(arXiv)](https://arxiv.org/abs/2312.07691)[(published)](https://doi.org/10.1038/s41534-024-00916-8). So it does not include ADAPT-VQE-GCM. 

It requires two python files (not need to install, but just copy the files) in the [original ADAPT-VQE repo](https://github.com/mayhallgroup/adapt-vqe). Those files are used for generating Hamiltonians and ansatz pools, so we can guarantee the performance comparision between ADAPT-VQE and ADAPT-GCIM is accurate in code level. We highly appreciate the accessibility of the source code of ADAPT-VQE. 

Due to package upgrades, updates on python imports in those external files must be made to run with the later version of `pyscf` and `openfermion`. We provide the updated version in [our forked repository](https://github.com/Firepanda415/adapt-vqe-for-gcim/tree/master).


## Applications on Downfolded Hamiltonians

In our recent paper [arXiv](https://arxiv.org/abs/2507.01199), we apply ADAPT-GCIM to downfolded Hamiltonians. Please visit https://github.com/npbauman/DUCC-Hamiltonian-Library for downfolded Hamiltonians and all the related experiments. The updated ADAPT-GCIM code is also contained in `Quantum_experiments/ADAPT-GCIM/src` folder in that repository.
