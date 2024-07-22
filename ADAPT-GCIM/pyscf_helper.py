import numpy as np
import scipy 
from pyscf import gto, scf, mcscf, fci, ao2mo, lo,  cc
from pyscf.tools import molden
from pyscf.cc import ccsd
import copy as cp

from sys import getsizeof
import h5py
import pyscf
from pyscf import lib

from functools import reduce

import openfermion 




class SQ_Hamiltonian:
    """
    General hamiltonian operator: 
    H = H(pq){p'q} + V(pqrs){p'q'sr} + A(p){p'} + B(p){p} + C(pq){p'q'} + D(pq){pq} + ...
    """
    def __init__(self):
        # operator integrals tensors
        self.int_H = np.array(())
        self.int_V = np.array(())
        self.int_A = np.array(())
        self.int_B = np.array(())
        self.int_C = np.array(())
        self.int_D = np.array(())

        # MO basis : really, just any transformation from AO->Current basis such that C'SC = I
        self.C = np.array(())
                            
        # AO overlap        
        self.S = np.array(())

        self.n_orb = 0

    def init(self, h, v, C, S):
        # molecule is a pyscf molecule object from gto.Mole()
        
        self.S = cp.deepcopy(S) 
        self.C = cp.deepcopy(C) 
        self.n_orb = self.C.shape[1]
        
        self.int_H = cp.deepcopy(h) 
        self.int_V = cp.deepcopy(v) 

    def transform_orbitals(self, U):
        """
        A(p,q)U(q,s) with s is new basis
        """
        assert(U.shape[0] == self.n_orb)

        new_dim = U.shape[1]

        print(" Rotate orbitals")
        self.int_H = np.einsum("pq,pr->rq",self.int_H,U)
        self.int_H = np.einsum("rq,qs->rs",self.int_H,U)
        self.int_V = np.einsum("pqrs,pl->lqrs",self.int_V,U)
        self.int_V = np.einsum("lqrs,qm->lmrs",self.int_V,U)
        self.int_V = np.einsum("lmrs,rn->lmns",self.int_V,U)
        self.int_V = np.einsum("lmns,so->lmno",self.int_V,U)

        if self.int_A.size > 0: 
            self.int_A = np.einsum(' r,rs->s', self.int_A,U) 
        if self.int_B.size > 0: 
            self.int_B = np.einsum(' r,rs->s', self.int_B,U) 
        if self.int_C.size > 0: 
            self.int_C = np.einsum(' pq,qr,rs->ps', U.T,self.int_C,U) 
        if self.int_D.size > 0: 
            self.int_D = np.einsum(' pq,qr,rs->ps', U.T,self.int_D,U) 

        
        self.C = self.C.dot(U)
        self.n_orb = self.C.shape[1]


    def energy_of_determinant(self, config_a, config_b):
        """ This only returns electronic energy"""
        e1 = 0
        e2 = 0
        for i in config_a:
            e1 += self.int_H[i,i]
        for i in config_b:
            e1 += self.int_H[i,i]
        for i in config_a:
            for j in config_a:
                if i>=j:
                    continue
                e2 += self.int_V[i,i,j,j]
                e2 -= self.int_V[i,j,j,i]
        for i in config_b:             
            for j in config_b:         
                if i>=j:                
                    continue           
                e2 += self.int_V[i,i,j,j]
                e2 -= self.int_V[i,j,j,i]
        for i in config_a:             
            for j in config_b:         
                e2 += self.int_V[i,i,j,j]
        e = e1+e2
        return e


    def extract_local_hamiltonian(self,orb_subset):
        """ Extract local Hamiltonian acting only on subset of orbitals """
        assert(len(orb_subset) <= self.n_orb)
        
        H = SQ_Hamiltonian()
        H.C = self.C[:,orb_subset]
        H.n_orb = H.C.shape[1]
        H.int_H = self.int_H[:,orb_subset][orb_subset]
        H.int_V = self.int_V[:,:,:,orb_subset][:,:,orb_subset][:,orb_subset][orb_subset]

        # Note: this doesn't pass other terms - can't see why it'd need to

        return H

    def export_FermionOperator(self, shift=0):

        """
        We have spatial orbital integrals, so we need to convert back to spin orbitals
        """
        fermi_op = openfermion.FermionOperator()

        #H
        for p in range(self.int_H.shape[0]):
            pa = 2*p + shift
            pb = 2*p+1 +  shift
            for q in range(self.int_H.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), self.int_H[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), self.int_H[p,q]) 
        
        #V
        for p in range(self.int_V.shape[0]):
            pa = 2*p +shift
            pb = 2*p+1 +shift
            for q in range(self.int_V.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                for r in range(self.int_V.shape[2]):
                    ra = 2*r +shift
                    rb = 2*r+1 +shift
                    for s in range(self.int_V.shape[3]):
                        sa = 2*s +shift
                        sb = 2*s+1 +shift
                        #aa
                        fermi_op += .5* openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ab
                        fermi_op += .5*openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ba
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qa,1),(sa,0),(rb,0)), self.int_V[p,r,q,s]) 
                        #bb
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), self.int_V[p,r,q,s]) 

        #A
        for p in range(self.int_A.shape[0]):
            pa = 2*p
            pb = 2*p+1
            fermi_op += openfermion.FermionOperator(((pa,1)), self.int_A[p]) 
            fermi_op += openfermion.FermionOperator(((pb,1)), self.int_A[p]) 
    
        #B
        for p in range(self.int_B.shape[0]):
            pa = 2*p
            pb = 2*p+1
            fermi_op += openfermion.FermionOperator(((pa,0)), self.int_B[p]) 
            fermi_op += openfermion.FermionOperator(((pb,0)), self.int_B[p]) 
        
        #C
        for p in range(self.int_C.shape[0]):
            pa = 2*p
            pb = 2*p+1
            for q in range(self.int_C.shape[1]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pa,1),(qb,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qa,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,1)), self.int_C[p,q]) 
        
        #D
        for p in range(self.int_D.shape[0]):
            pa = 2*p
            pb = 2*p+1
            for q in range(self.int_D.shape[1]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,0),(qa,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pa,0),(qb,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,0),(qa,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,0),(qb,0)), self.int_D[p,q]) 
        


        return fermi_op











def init(molecule,charge,spin,basis,n_frzn_occ=0, n_act=None, mo_order=None):
# {{{
    #PYSCF inputs
    print(" ---------------------------------------------------------")
    print("                                                          ")
    print("                      Using Pyscf:")
    print("                                                          ")
    print(" ---------------------------------------------------------")
    print("                                                          ")
    mol = gto.Mole()
    mol.atom = molecule

    # this is needed to prevent openblas - openmp clash for some reason
    # todo: take out
    lib.num_threads(1)

    mol.max_memory = 1000 # MB
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    mol.build()

    #orbitals and electrons
    n_orb = mol.nao_nr()
    n_b , n_a = mol.nelec 
    nel = n_a + n_b

    if n_act == None:
        n_act = n_orb
    #SCF 
    mf = scf.RHF(mol).run()
    #mf = scf.ROHF(mol).run()
    
    if mo_order != None:
        print(len(mo_order) , mf.mo_coeff.shape[1])
        assert(len(mo_order) == mf.mo_coeff.shape[1])
        mf.mo_coeff = mf.mo_coeff[:,mo_order]
    
    C = mf.mo_coeff #MO coeffs
    S = mf.get_ovlp()

    
    # dump orbitals for viewing 
    molden.from_mo(mol, 'orbitals_canon.molden', C)

    ##READING INTEGRALS FROM PYSCF
    E_nuc = gto.Mole.energy_nuc(mol)
    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph') 
    hcore = T + V
    S = mol.intor('int1e_ovlp_sph')
    g = mol.intor('int2e_sph')

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" %(mol.basis))
    print("Number of Orbitals                             :%10i" %(n_orb))
    print("Number of electrons                            :%10i" %(nel))
    print("Nuclear Repulsion                              :%16.10f " %E_nuc)
    print("Electronic SCF energy                          :%16.10f " %(mf.e_tot-E_nuc))
    print("SCF Energy                                     :%16.10f"%(mf.e_tot))


    print(" AO->MO")
    g = np.einsum("pqrs,pl->lqrs",g,C)
    g = np.einsum("lqrs,qm->lmrs",g,C)
    g = np.einsum("lmrs,rn->lmns",g,C)
    g = np.einsum("lmns,so->lmno",g,C)


    assert(n_frzn_occ <= n_b)
    n_frzn_vir = n_orb - n_act - n_frzn_occ
    assert(n_frzn_vir >= 0)

    n_a   -= n_frzn_occ
    n_b   -= n_frzn_occ
    n_orb -= n_frzn_occ

    print(" NElectrons: %4i %4i" %(n_a, n_b))
    Cact = C[:,n_frzn_occ:n_frzn_occ+n_act]
    Cocc = C[:,0:n_frzn_occ]

    dm = Cocc @ Cocc.T
    j, k = scf.hf.get_jk(mol, dm)
 
    t = hcore + 2*j - k
    h = reduce(np.dot, (Cact.conj().T, hcore + 2*j - k, Cact))
    ecore = np.trace(2*dm @ (hcore + j - .5*k))
    print(" ecore: %12.8f" %ecore)

    E_nuc += ecore
    def view(h5file, dataname='eri_mo'):
        f5 = h5py.File(h5file)
        print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
        f5.close()
    
    eri_act = ao2mo.outcore.general_iofree(mol, (Cact,Cact,Cact,Cact), intor='int2e', aosym='s4', compact=True)
    #view('ints_occ.h5')
    #view('ints_act.h5')
    eri_act = ao2mo.restore('s1', eri_act, Cact.shape[1])
    print(" ERIs in the active-space:")
    print(eri_act.shape, " %12.8f Mb" %(eri_act.nbytes*1e-6))

    if False:
        #compute slater determinant energy
        e1 = 0
        e2 = 0
        config_a = range(n_a)
        config_b = range(n_b)
        print(config_a,config_b)
        for i in config_a:
            e1 += h[i,i]
        for i in config_b:
            e1 += h[i,i]
        for i in config_a:
            for j in config_a:
                if i>=j:
                    continue
                e2 += eri_act[i,i,j,j]
                e2 -= eri_act[i,j,j,i]
        for i in config_b:             
            for j in config_b:         
                if i>=j:                
                    continue           
                e2 += eri_act[i,i,j,j]
                e2 -= eri_act[i,j,j,i]
        for i in config_a:             
            for j in config_b:         
                e2 += eri_act[i,i,j,j]
        e = e1+e2
        print("*HF Energy: %12.8f" %(e+E_nuc))
    
    fci = 0
    #pyscf FCI
    if fci:
        print()
        print(" ----------------------")
        print(" PYSCF")
        mc = mcscf.CASCI(mf, n_act, (n_a,n_b),ncore=n_frzn_occ)
        #mc.fcisolver = pyscf.fci.solver(mf, singlet=True)
        #mc.fcisolver = pyscf.fci.direct_spin1.FCISolver(mol)
        efci, ci = mc.kernel()
        print(" PYSCF: FCI energy: %12.8f" %(efci))
        print()
    
    Cact = C[:,n_frzn_occ:n_frzn_occ+n_act]
    return(n_act, n_a, n_b, h, eri_act, mol, E_nuc ,mf.e_tot,Cact,S)
# }}}





def init_pyscf(molecule,charge,spin,basis,orbitals):
# {{{
    #PYSCF inputs
    print(" ---------------------------------------------------------")
    print("                                                          ")
    print("                      Using Pyscf:")
    print("                                                          ")
    print(" ---------------------------------------------------------")
    print("                                                          ")
    mol = gto.Mole()
    mol.atom = molecule

    # this is needed to prevent openblas - openmp clash for some reason
    # todo: take out
    lib.num_threads(1)

    mol.max_memory = 1000 # MB
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    mol.build()

    #orbitals and electrons
    n_orb = mol.nao_nr()
    n_b , n_a = mol.nelec 
    nel = n_a + n_b

    #SCF 
    mf = scf.RHF(mol).run()
    #mf = scf.ROHF(mol).run()
    C = mf.mo_coeff #MO coeffs
    S = mf.get_ovlp()

    print(" Orbs1:")
    print(C)
    Cl = cp.deepcopy(C)
    if orbitals == "boys":
        print("\nUsing Boys localised orbitals:\n")
        cl_o = lo.Boys(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.Boys(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))

    elif orbitals == "pipek":
        print("\nUsing Pipek-Mezey localised orbitals:\n")
        cl_o = lo.PM(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.PM(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))
    elif orbitals == "edmiston":
        print("\nUsing Edmiston-Ruedenberg localised orbitals:\n")
        cl_o = lo.ER(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.ER(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))
#    elif orbitals == "svd":
#        print("\nUsing SVD localised orbitals:\n")
#        cl_o = cp.deepcopy(mf.mo_coeff[:,:n_a])
#        cl_v = cp.deepcopy(mf.mo_coeff[:,n_a:])
#
#        [U,s,V] = np.linalg.svd(cl_o)
#        cl_o = cl_o.dot(V.T)
#        Cl = np.column_stack((cl_o,cl_v))
    elif orbitals == "canonical":
        print("\nUsing Canonical orbitals:\n")
        pass
    else:
        print("Error: Wrong orbital specification:")
        exit()

    
    print(" Overlap:")
    print(C.T.dot(S).dot(Cl))

    # sort by cluster
    blocks = [[0,1,2,3],[4,5,6,7]]
    O = Cl[:,:n_a] 
    V = Cl[:,n_a:] 
    [sorted_order, cluster_sizes] = mulliken_clustering(blocks,mol,O)
    O = O[:,sorted_order]
    [sorted_order, cluster_sizes] = mulliken_clustering(blocks,mol,V)
    V = V[:,sorted_order]
    Cl = np.column_stack((O,V)) 
    
    C = cp.deepcopy(Cl)
    # dump orbitals for viewing 
    molden.from_mo(mol, 'orbitals_canon.molden', C)
    molden.from_mo(mol, 'orbitals_local.molden', Cl)

    ##READING INTEGRALS FROM PYSCF
    E_nu = gto.Mole.energy_nuc(mol)
    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph') 
    hcore = T + V
    S = mol.intor('int1e_ovlp_sph')
    g = mol.intor('int2e_sph')

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" %(mol.basis))
    print("Number of Orbitals                             :%10i" %(n_orb))
    print("Number of electrons                            :%10i" %(nel))
    print("Nuclear Repulsion                              :%16.10f " %E_nu)
    print("Electronic SCF energy                          :%16.10f " %(mf.e_tot-E_nu))
    print("SCF Energy                                     :%16.10f"%(mf.e_tot))


    print(" AO->MO")
    g = np.einsum("pqrs,pl->lqrs",g,C)
    g = np.einsum("lqrs,qm->lmrs",g,C)
    g = np.einsum("lmrs,rn->lmns",g,C)
    g = np.einsum("lmns,so->lmno",g,C)

    h = reduce(np.dot, (C.conj().T, hcore, C))

    
#    #mf = mf.density_fit(auxbasis='weigend')
#    #mf._eri = None
#    mcc = cc.UCCSD(mf)
#    eris = mcc.ao2mo()
#    eris.g = g
#    eris.focka = h
#    eris.fockb = h
#    
#    emp2, t1, t2 = mcc.init_amps(eris)
#    exit()
#    print(abs(t2).sum() - 4.9318753386922278)
#    print(emp2 - -0.20401737899811551)
#    t1, t2 = update_amps(mcc, t1, t2, eris)
#    print(abs(t1).sum() - 0.046961325647584914)
#    print(abs(t2).sum() - 5.378260578551683   )
#    
#    
#    exit()

    return(n_orb, n_a, n_b, h, g, mol, E_nu,mf.e_tot,C,S)
# }}}





def mulliken_clustering(clusters,mol,C):
    aos_per_atom = []
    aos_per_cluster = []

    #check n atoms
    max_atomid = 0
    tmp_natm = 0
    for c in clusters:
        for ci in c:
            max_atomid = max(max_atomid,ci)
            tmp_natm += 1

    assert(max_atomid == tmp_natm-1)
    assert(tmp_natm == mol.natm)
    #for d in dir(mol):
    #    print(d)
    for a in range(mol.natm):
        aos_per_atom.append([])
    for mu in range(C.shape[0]):
        aos_per_atom[mol.bas_atom(mu)].append(mu)

    mos_per_cluster = [[] for i in range(len(clusters))]
    print(" AOs per cluster")
    for c in clusters:
        tmp = []
        for ci in c:
            tmp.extend(aos_per_atom[ci])
        aos_per_cluster.append(tmp)
        print(tmp)

    S = mol.intor('int1e_ovlp_sph')
    for i in range(C.shape[1]):
        print(" Orbital %4i" %i)
        pi = np.einsum('a,b,bc->ac',C[:,i],C[:,i],S)
        
        pop_per_cluster = []

        for c in aos_per_cluster:
            pop_per_cluster.append(np.trace(pi[:,c][c]))
        mos_per_cluster[np.argmax(pop_per_cluster)].append(i)
    print(mos_per_cluster)
    sorted_orb_list = []
    cluster_sizes = []
    for c in mos_per_cluster:
        sorted_orb_list.extend(c)
        cluster_sizes.append(len(c))

    return(sorted_orb_list,cluster_sizes)



    



