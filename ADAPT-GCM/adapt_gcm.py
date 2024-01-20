import numpy as np
import scipy
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as ssl
import sys

import openfermion as of
from openfermion import *


## ------------------- Eigenvalue computation while solving the numerical issues Starts -----------------------
def gcm_eig_helper(Hmat, Smat):
    try:
        evals_def = sl.eigh(Hmat, Smat)[0]
        return evals_def, True
    except:
        return None, False

def gcm_eig(H_indef, S_indef, ev_thresh=1e-15, printout=True):
    evals, evecs = sl.eig(S_indef)
    # sort from largest to smallest
    idx = evals.argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]

    solvable_flag = False
    curr_thresh = ev_thresh
    while not solvable_flag:
        num_pos_eigs = np.sum(np.abs(evals) > curr_thresh)
        evecs_trunc = evecs[:,:num_pos_eigs]
        evecs_trunc = np.matrix(evecs_trunc)
        S_def = evecs_trunc.H.dot(S_indef).dot(evecs_trunc)
        H_def = evecs_trunc.H.dot(H_indef).dot(evecs_trunc)
        #
        evals_def, solvable_flag = gcm_eig_helper(H_def,S_def)
        if not solvable_flag:
            curr_thresh += ev_thresh
    if printout:
        print("   Eigensolver: dimension reduced from {:d} to {:d}, with eigval > {:.3e}".format(len(evals), num_pos_eigs, curr_thresh))
    return evals_def




## ---------------------------------------------------------

def gcm_grad(op_expmat,bas_old,hf,Hmat):
    #
    # _,op = ans_pool(indx_tuple,rot)
    bas = bas_old.copy()
    bas.insert(0, sps.csc_matrix( np.eye(op_expmat.shape[0]) ) )
    bas.append(op_expmat)
    #
    L = len(bas)
    Heff = np.zeros((L,L),dtype=complex)
    ovlp = np.zeros((L,L),dtype=complex)
    d_Heff = np.zeros((L,L),dtype=complex)
    d_ovlp = np.zeros((L,L),dtype=complex)
    for i in range(L):
        for j in range(L):
            _ = sps.csc_matrix.conj(bas[i]@hf)
            _ = sps.csc_matrix.transpose(_)
            Heff[i,j] = _.dot(Hmat.dot(bas[j]@hf))[0,0].real
            ovlp[i,j] = _.dot(bas[j]@hf)[0,0].real
            if i == L-1 and j != L-1:
                d_Heff[i,j] = -1*_.dot((bas[i]@Hmat).dot(bas[j]@hf))[0,0].real
                d_ovlp[i,j] = _.dot(bas[i].dot(bas[j]@hf))[0,0].real
            elif i != L-1 and j == L-1:
                d_Heff[i,j] = _.dot((Hmat@bas[i]).dot(bas[j]@hf))[0,0].real
                d_ovlp[i,j] = _.dot(bas[i].dot(bas[j]@hf))[0,0].real
            elif i != L-1 and j != L-1:
                d_Heff[i,j] = _.dot((Hmat@bas[i]-bas[i]@Hmat).dot(bas[j]@hf))[0,0].real
    e,f = sl.eig(Heff,ovlp)
    idx = np.argsort(e)
    f = f[:,idx]
    H_k = np.conj(f[0]).T@Heff@f[0]
    S_k = np.conj(f[0]).T@ovlp@f[0]
    d_H_k = np.conj(f[0]).T@d_Heff@f[0]
    d_S_k = np.conj(f[0]).T@d_ovlp@f[0]
    g = (d_H_k*S_k - H_k*d_S_k)/(S_k*S_k)
    return g

# No +- for basis
def gcm_mats_forag(ham_mat, GCM_SINGLE_MATs, HF_state, iter_coeffs, 
                   constant_t = np.pi/4,
                    prev_basis = None,
                    make_orth = False):
    """Create H and S matrices from GCM methods
        The difference between gcm_mats() is the GCM_SINGLE_MATs in this function has the lastest 
        basis matrix at the END of the list

    Args:
        ham_mat (_type_): _description_
        GCM_SINGLE_MATs (_type_): _description_
        HF_state (_type_): _description_
        iter_coeffs (_type_): _description_
        make_orth (boolean): If orthogonalize the GCM basis so overlap matrix S is just an identity matrix. Default is False
        prev_basis : to pass necessary basis vectors from last iteration
    """
    ### Construct Basis
    num_basis = len(GCM_SINGLE_MATs)
    latest_basis_index = len(GCM_SINGLE_MATs) - 1
    GCM_BASIS = {'R0': HF_state.copy()}

    for k in range(len(GCM_SINGLE_MATs)):
        t = iter_coeffs[k] if iter_coeffs[k] != 0 else constant_t
        opmat = GCM_SINGLE_MATs[k]
        GCM_BASIS['+R'+str(k)] = ssl.expm_multiply(   (t*opmat), HF_state  )

    # Final Product basis, only +/- in the newly-added element
    if len(GCM_SINGLE_MATs) > 1:
        prod_basis = HF_state.copy()
        for i in range( len(GCM_SINGLE_MATs) ):
            op = GCM_SINGLE_MATs[i]
            op_coef = iter_coeffs[i]
            prod_basis = ssl.expm_multiply(  (op_coef*op), prod_basis  )
        GCM_BASIS['+R'+str( len(GCM_SINGLE_MATs) )] = prod_basis

        if not make_orth:
            # Add product bases from previous iterations
            ri = 1
            bi = 0
            while bi < len(prev_basis):
                GCM_BASIS['+R'+str(num_basis+ri)] = prev_basis[bi]
                ri += 1
                bi += 1
            # Only the new product bases are carrid out to the next iteration
            prev_basis.append(GCM_BASIS['+R'+str(num_basis)])
    
    ### Compute Matrices
    names = list(GCM_BASIS.keys())
    if not make_orth: # Usual GCM method
        H = np.zeros((len(names),len(names)), dtype=complex)
        S = np.zeros((len(names),len(names)), dtype=complex)
        basis_mat = np.zeros(   (GCM_BASIS[names[0]].shape[0], len(names))   , dtype=complex)
        for i in range(len(names)):
            basis_mat[:,i] = GCM_BASIS[ names[i] ].toarray().flatten()
            for j in range(i,len(names)):
                left_op  = GCM_BASIS[ names[i] ].transpose().conj()
                right_op = GCM_BASIS[ names[j] ]
                H[i,j] = left_op.dot(  ham_mat.dot(right_op)  )[0,0]
                S[i,j] = left_op.dot(right_op)[0,0]
                if i != j:
                    H[j,i] = H[i,j].conj()
                    S[j,i] = S[i,j].conj()
    else: # Use orthogonalization
        # Create a matrix whose column is each GCM basis
        if len(GCM_SINGLE_MATs) > 1:
            mat_height = prev_basis.shape[0]
            num_orthbasis = prev_basis.shape[1]+2
            basis_mat = np.zeros((mat_height, num_orthbasis), dtype=complex)
            basis_mat[:,:num_orthbasis-2] = prev_basis
            basis_mat[:,num_orthbasis-2] = GCM_BASIS['+R'+str(latest_basis_index)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            basis_mat[:,num_orthbasis-1] = GCM_BASIS['+R'+str(num_basis)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
        else:
            mat_height = prev_basis.shape[0]
            num_orthbasis = prev_basis.shape[1]+1
            basis_mat = np.zeros((mat_height, num_orthbasis), dtype=complex)
            basis_mat[:,:num_orthbasis-1] = prev_basis
            basis_mat[:,num_orthbasis-1] = GCM_BASIS['+R'+str(latest_basis_index)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix

        orth_basis_mat = sl.orth(basis_mat)
        H = np.zeros((orth_basis_mat.shape[1],orth_basis_mat.shape[1]), dtype=complex)
        S = np.zeros((orth_basis_mat.shape[1],orth_basis_mat.shape[1]), dtype=complex) # We can check if S is identity
        # Create H from orthogonalized matrix
        for p in range(orth_basis_mat.shape[1]):
            for q in range(p, orth_basis_mat.shape[1]):
                left_op  = orth_basis_mat[:,p].transpose().conj()
                right_op = orth_basis_mat[:,q]
                H[p,q] = left_op.dot(  ham_mat.dot(right_op)  )
                S[p,q] = left_op.dot(right_op)
                if p != q:
                    H[q,p] = H[p,q].conj()
                    S[p,q] = S[p,q].conj()
        print(" S is an identity matrix:", np.allclose(S, np.identity(S.shape[0]))) # for verfication

    if np.linalg.norm(H.imag) < 1e-14:
        H = H.real
    if np.linalg.norm(S.imag) < 1e-14:
        S = S.real

    if not make_orth:
        return H, S, prev_basis
    else:
        return H, S, orth_basis_mat
    

# One-shot GCM, only +
def gcm_mats_forag_OSP(ham_mat, GCM_SINGLE_MATs, HF_state, iter_coeffs, 
                   constant_t = np.pi/4,
                    prev_basis = None,
                    make_orth = False):
    """Create H and S matrices from GCM methods
        The difference between gcm_mats() is the GCM_SINGLE_MATs in this function has the lastest 
        basis matrix at the END of the list

    Args:
        ham_mat (_type_): _description_
        GCM_SINGLE_MATs (_type_): _description_
        HF_state (_type_): _description_
        iter_coeffs (_type_): _description_
        make_orth (boolean): If orthogonalize the GCM basis so overlap matrix S is just an identity matrix. Default is False
        prev_basis : to pass necessary basis vectors from last iteration
    """
    num_basis = len(GCM_SINGLE_MATs)
    latest_basis_index = len(GCM_SINGLE_MATs) - 1
    GCM_BASIS = {'R0': HF_state.copy()}
    # constant_t = np.pi/4
    print(" Orthogonalization: ", make_orth)

    for k in range(len(GCM_SINGLE_MATs)):
        t = iter_coeffs[k] if iter_coeffs[k] != 0 else constant_t
        opmat = GCM_SINGLE_MATs[k]
        GCM_BASIS['+R'+str(k)] = ssl.expm_multiply(   (t*opmat), HF_state  )

    # Final Product basis, only +/- in the newly-added element
    if len(GCM_SINGLE_MATs) > 1:
        prod_basis = HF_state.copy()
        for i in range( len(GCM_SINGLE_MATs) ):
            op = GCM_SINGLE_MATs[i]
            op_coef = iter_coeffs[i]
            prod_basis = ssl.expm_multiply(  (op_coef*op), prod_basis  )
        GCM_BASIS['+R'+str( len(GCM_SINGLE_MATs) )] = prod_basis
    
    ### Compute Matrices
    names = list(GCM_BASIS.keys())
    if not make_orth: # Usual GCM method
        H = np.zeros((len(names),len(names)), dtype=complex)
        S = np.zeros((len(names),len(names)), dtype=complex)
        basis_mat = np.zeros(   (GCM_BASIS[names[0]].shape[0], len(names))   , dtype=complex)
        for i in range(len(names)):
            basis_mat[:,i] = GCM_BASIS[ names[i] ].toarray().flatten()
            for j in range(i,len(names)):
                left_op  = GCM_BASIS[ names[i] ].transpose().conj()
                right_op = GCM_BASIS[ names[j] ]
                H[i,j] = left_op.dot(  ham_mat.dot(right_op)  )[0,0]
                S[i,j] = left_op.dot(right_op)[0,0]
                if i != j:
                    H[j,i] = H[i,j].conj()
                    S[j,i] = S[i,j].conj()
    else: # Use orthogonalization
        basis_mat = np.zeros(   (GCM_BASIS[names[0]].shape[0], len(names))   , dtype=complex)
        for ban_index in range(len(names)):
            basis_mat[:,ban_index] = GCM_BASIS[ names[ban_index] ].toarray().flatten()

        orth_basis_mat = sl.orth(basis_mat)
        # S = np.identity(orth_basis_mat.shape[1]) # trivial
        H = np.zeros((orth_basis_mat.shape[1],orth_basis_mat.shape[1]), dtype=complex)
        S = np.zeros((orth_basis_mat.shape[1],orth_basis_mat.shape[1]), dtype=complex) # We can check if S is identity
        # Create H from orthogonalized matrix
        for p in range(orth_basis_mat.shape[1]):
            for q in range(p, orth_basis_mat.shape[1]):
                left_op  = orth_basis_mat[:,p].transpose().conj()
                right_op = orth_basis_mat[:,q]
                H[p,q] = left_op.dot(  ham_mat.dot(right_op)  )
                S[p,q] = left_op.dot(right_op)
                if p != q:
                    H[q,p] = H[p,q].conj()
                    S[p,q] = S[p,q].conj()
        print(" S is an identity matrix:", np.allclose(S, np.identity(S.shape[0]))) # for verfication

    if np.linalg.norm(H.imag) < 1e-14:
        H = H.real
    if np.linalg.norm(S.imag) < 1e-14:
        S = S.real

    if not make_orth:
        return H, S, prev_basis
    else:
        return H, S, orth_basis_mat



## ADAPT-GCM major function
def adapt_gcm(ham_op, ansatz_pool, HF_state, theta, 
                change_tol    = 1e-6,
                max_iter  = 200,
                file_prefix    = None,
                make_orth = False,
                use_gcmgrad  = False,
                ):
    """Running ADAPT GCM at the same time

    Args:
        change_tol (float): if lowest eigenvalue in a certain number of itertions are changed within change_tol, then converges
        file_prefix (str): file path for saving GCM matrices in each iteration END with '/'. None for not saving matrices. If path does not exists, it does NOT create the path. Defaults to None. 
        make_orth (bool, optional): If make basis orthgonal when construct matrices for GCM. Defaults to False. 
    """
    ansatz_pool.generate_SparseMatrix()
    ansatz_pool.gradient_print_thresh = 100 # No print
    ham_mat = of.linalg.get_sparse_operator(ham_op)
    # ref_energy = HF_state.T.conj().dot(ham_mat.dot(HF_state))[0,0].real
    # print(" Reference Energy: %12.8f" %ref_energy)

    long_flat_counter = 0 # Counter for how many iterations the eigenvalues does not have large changes
    GCM_DIFFS = [] # Error from FCI energy for GCM algorihtm
    max_iter = min(max_iter, len(ansatz_pool.fermi_ops))
    parameters = [] # Currently, it is trivally all pi/4, may have some change in the future to reduce numerical instability
    ansatz_ops = []     # Ansatz operator objects
    ansatz_mat = []     # Ansatz operator in scipy sparse matrix form
    ansatz_expmat = []  # Matrix expoential needed for computing GCM gradient
    ansatz_indices = [] # Record the selected ansatz indices to avoid dupliction
    GCM_RESULT = [0]    # Record eigenvalues from ONE iteration of GCM
    ev_records = [0]    # Record lowest eigenvalues in every ADAPT iteration
    GCM_BASIS_SIZE = [] # Record number of bases in every ADAPT iteration
    prodBasis_set = []  # A set to record product bases in all previous iterations

    vqe_curr_vec = HF_state.copy()
    flat_ref_ket = HF_state.toarray().flatten()
    last_basis_mat = flat_ref_ket.reshape(  (len(flat_ref_ket), 1)  )
    true_lowest_ev = ssl.eigsh(ham_mat,1,which='SA')[0][0].real # Muqing: Find the actual FCI energy
    
    sig = ham_mat.dot(HF_state)
    for n_iter in range(0,max_iter):
        print("\n\n------------ Iter {:d} Start ------------".format(n_iter))
        next_index = None
        next_deriv = 0
        next_expop = None
        norm_sum = 0

        sig = ham_mat.dot(vqe_curr_vec)
        for oi in range(ansatz_pool.n_ops):
            if oi not in ansatz_indices:
                if use_gcmgrad:
                    op_expmat = ssl.expm( theta*ansatz_pool.spmat_ops[oi] )
                    gi = gcm_grad(op_expmat,ansatz_expmat,vqe_curr_vec,ham_mat)
                    norm_sum += gi*gi
                else:
                    gi = ansatz_pool.compute_gradient_i(oi, vqe_curr_vec, sig)
                    norm_sum += gi*gi
                if abs(gi) > abs(next_deriv):
                    next_deriv = gi
                    next_index = oi

        norm_sum = np.sqrt(norm_sum)
        print(" Sum of gradients = {:8.6f}".format(norm_sum) )
        print(" Largest gradient = {:8.6f}".format(abs(next_deriv)) )

        ## Check convergence
        converged = False
        if ansatz_pool.n_ops <= len(ansatz_indices):
            print(" Converged due to running out the operators")
            converged = True
        if norm_sum <= 1e-8 or next_index is None:
            print(" Converged due to all gradients are 0")
            converged = True

        flat_iter = max(10, int(0.1*np.abs(ansatz_pool.n_ops - len(ansatz_indices))) )
        if len(ev_records) > min(4, len(ansatz_pool.spmat_ops)): # no check in first 3 iterations to avoid "quick" converges
            if (np.abs(ev_records[-1] - ev_records[-2])) < change_tol: ## if absolute change is too smal
                print(" Small changes have happended for {:d}/{:d} iterations".format(long_flat_counter, flat_iter))
                long_flat_counter += 1
                if long_flat_counter >= flat_iter:
                    print(" Converged due to tiny changes {:.3e} in eigenvalues in {:d} iterations".format( np.abs(ev_records[-1] - ev_records[-2]), flat_iter))
                    converged = True
            else:
                long_flat_counter = 0 # reset the counter

        ## If converged
        if converged:
            print(" ADAPT-GCM Converged")
            # One-shot GCM
            osgcmHP, osgcmSP, _ = gcm_mats_forag_OSP(ham_mat, ansatz_mat, HF_state, parameters,
                                                        constant_t=theta,
                                                        prev_basis = prodBasis_set, make_orth = make_orth)
            if not make_orth:
                osgcm_resultP = gcm_eig(osgcmHP, osgcmSP, ev_thresh=1e-14, printout=True)
            else:
                osgcm_resultP = sl.eigh(osgcmHP)[0]
            print(" Number of ansatzes: ", len(ansatz_ops))
            print(" Number of basis used in the final iteration of GCM:", GCM_BASIS_SIZE[-1])
            print(" Number of basis used in the one-shot GCM :", osgcmHP.shape[0])
            print(" *True Energy : {:19.15f}".format(true_lowest_ev) )
            print(" *GCM Finished: {:19.15f}".format(GCM_RESULT[0]) )
            gcm_diff = np.abs(true_lowest_ev-GCM_RESULT[0])
            print(" *GCM error          {:8.6e}".format( gcm_diff ))
            # Report the converged result
            sys.stdout.flush() 
            osgcm_diffP = np.abs(true_lowest_ev-osgcm_resultP[0])
            print(" *One-shot GCM error {:8.6e}".format( osgcm_diffP ))
            # Print final ansatzes 
            print(" ----------- Final ansatzes ----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                opstring = ansatz_pool.get_string_for_term(ansatz_ops[si]) 
                print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
            print(" -------------------------------------- ")
            break
        
        ## If not converged
        print( "\n Choose ansatz {:d}".format(next_index) )
        parameters.append(theta)
        ansatz_ops.append(ansatz_pool.fermi_ops[next_index])
        ansatz_mat.append(ansatz_pool.spmat_ops[next_index])
        ansatz_indices.append(next_index)
        if use_gcmgrad:
            ansatz_expmat.append(next_expop)
        
        sys.stdout.flush() 
        ################ GCM Computation #######################
        if not make_orth:
            gcmH, gcmS, prodBasis_set = gcm_mats_forag(ham_mat, ansatz_mat, HF_state, parameters,
                                                        constant_t=theta,
                                                prev_basis = prodBasis_set, make_orth = make_orth)
            print(" Number of saved prod bases", len(prodBasis_set), ", H matrix size", gcmH.shape)
            GCM_BASIS_SIZE.append(gcmH.shape[0])
        else:
            gcmH, gcmS, last_basis_mat = gcm_mats_forag(ham_mat, ansatz_mat, HF_state, parameters, 
                                                        constant_t=theta,
                                                    prev_basis=last_basis_mat, make_orth = make_orth)
            print(" Basis matrix size", last_basis_mat.shape, ", H matrix size", gcmH.shape)
            GCM_BASIS_SIZE.append(gcmH.shape[0])
        if file_prefix is not None:
            try:
                np.save(file_prefix+'IterMats/IterAG{:d}_S.npy'.format(n_iter), gcmS)
                np.save(file_prefix+'IterMats/IterAG{:d}_H.npy'.format(n_iter), gcmH)
                np.save(file_prefix+'IterMats/IterAG{:d}_bases.npy'.format(n_iter), np.array(ansatz_mat))
                np.save(file_prefix+'IterMats/IterAG{:d}_preop_coeffs.npy'.format(n_iter), np.array(parameters))
            except:
                np.save(file_prefix+'IterAG{:d}_S.npy'.format(n_iter), gcmS)
                np.save(file_prefix+'IterAG{:d}_H.npy'.format(n_iter), gcmH)
                np.save(file_prefix+'IterAG{:d}_bases.npy'.format(n_iter), np.array(ansatz_mat))
                np.save(file_prefix+'IterAG{:d}_preop_coeffs.npy'.format(n_iter), np.array(parameters))
        if not make_orth:
            GCM_RESULT = gcm_eig(gcmH, gcmS, ev_thresh=1e-14, printout=True)
        else:
            GCM_RESULT = sl.eigh(gcmH)[0]
        ev_records.append(GCM_RESULT[0])
        ###################################################
        sys.stdout.flush()

        vqe_curr_vec = ssl.expm_multiply(theta*ansatz_pool.spmat_ops[next_index], vqe_curr_vec)
        gcm_diff = np.abs(true_lowest_ev-GCM_RESULT[0])
        GCM_DIFFS.append(gcm_diff)
        print(" GCM Energy: {:19.15f}".format( GCM_RESULT[0]))
        print(" GCM error : {:8.6e}".format( gcm_diff ))
        print("------------ Iter {:d} Finish -----------".format(n_iter))
    return ev_records[1:], osgcm_resultP[0], ansatz_indices, GCM_DIFFS, osgcm_diffP, GCM_BASIS_SIZE