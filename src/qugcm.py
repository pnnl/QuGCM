import numpy as np
import numpy.linalg as nl
from qiskit.opflow import PauliExpectation, PauliTrotterEvolution, MatrixExpectation
from qiskit.opflow import CircuitSampler, Suzuki, StateFn,CircuitStateFn
from qiskit.utils import QuantumInstance # https://qiskit.org/documentation/stubs/qiskit.utils.QuantumInstance.html
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit.opflow import X, Y, Z, I, PauliOp
import re # used in read basis index from e.g., '+R5-R6'
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import json # to dump Hamiltonian to a file

# Used in the notebook
import qiskit
import scipy.linalg as sl # eigh function for generalized eigenvalue problem
from numpy.random import Generator, PCG64
import time
from datetime import timedelta
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, execute  # used in notebook
# from qiskit.circuit.library import PauliEvolutionGate
# from qiskit.synthesis import SuzukiTrotter # https://arxiv.org/abs/quant-ph/0508139
from qiskit_nature.circuit.library import HartreeFock
# from distutils.command.config import config
from qiskit.providers.aer import AerSimulator,QasmSimulator # used in notebook




##--------------------------- String Processors for Hamiltonian file ------------------------##
def removeBlankLines(input_path, output_path):
    with open(input_path) as filehandle:
        lines = filehandle.readlines()

    with open(output_path, 'w') as filehandle:
        lines = filter(lambda x: x.strip(), lines)
        filehandle.writelines(lines)

def of2qisHamStr(ops_str_list, mapping, starting_index=0): # e.g, convert ['0^', '0'] to "+_0 -_0"
    s = ""
    for op in ops_str_list:
        if '^' in op:
            op = op.replace('^', '')
            s += '+_'
        else:
            s += '-_'
        op_reindex = str(int(op)-starting_index) # OpenFermion's starting_index
        op = mapping[op_reindex]
        s += op
        s += ' '
    return s[:-1] # remove the last space
    

def readHam(ham_file, num_spin_orbitals, mapping=None):
    first_time = True
    with open(ham_file, 'r') as filehandle:
        lines = filehandle.readlines()
        for line in lines:
            if len(line.strip())  == 0: # pass empty lines
                continue
            try:
                elements  = line.split('\'') # split creator annihilator string and coefficient (cannot use comma)
                # Extrate creation and annihilation operator
                operator_string = elements[1].lstrip() # to deal with the string like ' 9^ 11' with a space ahead
                ops_strings =operator_string.split()
                qis_ops = of2qisHamStr(ops_strings, mapping=mapping, starting_index=0)
                # Extrate coefficient
                if ',' in elements[-1]:
                    elements[-1] = elements[-1].split(',')[-1]
                coeff = float(elements[-1])
                # Form Qiskit FerionicOp
                if first_time:
                    H = FermionicOp((qis_ops, coeff),register_length=num_spin_orbitals,display_format='dense')
                    first_time = False
                else:
                    H += FermionicOp((qis_ops, coeff),register_length=num_spin_orbitals,display_format='dense')
            except:
                print(line, "is not working.")
    return H
##--------------------------- String Processors for Hamiltonian file ------------------------##





##--------------------------- String Processors for Hamiltonian file ------------------------##


## translate the basis configuration from OpenFermion and TCE format to Qiskit format
def of_to_qis_config(config_table, mapping, starting_index=0): # OpenFermion's starting_index
    """
    Transfer orbits in config_table from TCE conversion to Qiskit convension using provided mapping
    E.g., 
    If R1 in TCE is [('5^ 2',1), ('7^ 4',-1), ('2^ 5', -1), ('4^ 7',1)] (Note: indices start from 1)
    Then config_table could be
    basis_config_of = {
        'R1': [('5^ 2',1), ('7^ 4',-1), ('2^ 5', -1), ('4^ 7',1)],
        'R2': [('6^ 1',1), ('8^ 3',-1), ('1^ 6', -1), ('3^ 8',1)],
        'R3': [('6^ 2',1), ('8^ 4',-1), ('2^ 6', -1), ('4^ 8',1)],
        'R4': [('5^ 1',1), ('7^ 3',-1), ('1^ 5', -1), ('3^ 7',1)],
    } # only single operator, characters in keys must be capitalized
    This function will tranfer this python dictionary 
    using provided mapping 
    {'0': '0',
    '1': '1',
    '2': '4',
    '3': '5',
    '4': '2',
    '5': '3',
    '6': '6',
    '7': '7'}
    and starting index 1 to the following
    {'R1': [('+_2 -_1', 1), ('+_6 -_5', -1), ('+_1 -_2', -1), ('+_5 -_6', 1)],
    'R2': [('+_3 -_0', 1), ('+_7 -_4', -1), ('+_0 -_3', -1), ('+_4 -_7', 1)],
    'R3': [('+_3 -_1', 1), ('+_7 -_5', -1), ('+_1 -_3', -1), ('+_5 -_7', 1)],
    'R4': [('+_2 -_0', 1), ('+_6 -_4', -1), ('+_0 -_2', -1), ('+_4 -_6', 1)]}

    Args:
        config_table (dict): as shown in the summary
        mapping (dict): as shown in the summary {TCE orbit index: Qiskit orbit index}
        starting_index (int, optional): starting index of TCE convension's orbit index. Defaults to 0.

    Returns:
        dict: the confuguration table in Qiskit convension
    """
    new_config = {}
    for k,v in config_table.items():
        new_v = []
        for term in v:
            orb,coeff = term
            new_orb = of2qisHamStr(orb.split(), mapping, starting_index)
            new_v.append((new_orb, coeff))
        new_config[k] = new_v
    return new_config


def base_circuit_code_single(single_name, t, basis_dict, left):
    """ Construct a Fermonic string that corresponds to the input 'name'. E.g., 
    Translate '+R1' with t=0.1 to [('+_2 -_1', -0.1), ('+_6 -_5', 0.1), ('+_1 -_2', 0.1), ('+_5 -_6', -0.1)]
    If the operator is on the left side of the expectation, then flip the sign of t

    Parameters
    ----------
    name : string
        single: '<+ or ->R<number>' or composite: '<single><single>'. E.g., '+R1'. NOTE: do NOT deal with composite, e.g., '+R2-R1'. 
    ts : tuple (float,) or (float,float)
        parameter t for each single/composite operator. For practical purpose, some number in (0,0.1].
    basis_dict : dict
        configuration for basis in convention in Qiskit. e.g. basis_config_qis
    left : boolean
        if this operator is the left one (left one needs conjugate transpose)

    Returns
    -------
    Fermonic strings

    """
    ## For each single operator, construct the circuit
    ## and append to the previous circuit if necessary
    total_config = []
    sign_str = single_name[0] # '+' or '-'
    if sign_str == '+':
        if left: 
            t_act = -t
        else:
            t_act = t
    elif sign_str == '-':
        if left:
            t_act = t
        else:
            t_act = -t
    else:
        raise Exception('Unknow Sign')
    base_str = single_name[1:] #'R<num>', e.g., 'R1'
    ## Create Fermonic operator
    base_config = basis_dict[base_str] # obtain creation and annihilation operators information
    for term in base_config:
        total_config.append((term[0], term[1]*t_act))
    return total_config


def base_circuit_code_separate(name, ts, basis_dict, left):
    """ Construct a Fermonic string that corresponds to the input 'name'. E.g., 
    Translate '+R1' with t=0.1 to [('+_2 -_1', -0.1), ('+_6 -_5', 0.1), ('+_1 -_2', 0.1), ('+_5 -_6', -0.1)]
    The difference with the old base_circuit_code() method is this method deal with composites. For example, in the old method,
    '+R2-R1' is dealt as a whole like e^{+R2-R1}, which is not correct unless they are communte. 
    In this method, this becomes e^{+R2}e^{-R1} 

    Parameters
    ----------
    name : string
        single: '<+ or ->R<number>' or composite: '<single><single>'. E.g., '+R1'. NOTE: do NOT deal with composite, e.g., '+R2-R1'. 
    ts : tuple (float,) or (float,float)
        parameter t for each single/composite operator. For practical purpose, some number in (0,0.1].
    basis_dict : dict
        configuration for basis in convention in Qiskit. e.g. basis_config_qis
    left : boolean
        if this operator is the left one (left one needs conjugate transpose)

    Returns
    -------
    Fermonic strings

    """
    ## Detect single operator or a composite
    ## Better to use regexr to split the composite, for now just assume at most a double composite
    if_composite = name.count('R') - 1 
    name_list = []
    t_list = ts
    if if_composite: # if composite 
        mid_pos = name.rfind('R')-1 # e.g, for '+R2+R1', mid_pos=3, name1='+R1', name2='+R2'
        name0 = name[:mid_pos]
        name1 = name[mid_pos:]
        if left: # Note that ts are flipped in est_S_mat_qwc() and est_H_mat_qwc(). The minus sign will be added in base_circuit_code_single()
            name_list = [name1, name0] # if this operator is on the left, then this is for <\psi_i|, 
            #                            # conjugate transpose with an anti-Hermitian matrix becomes a negative sign
        else:
            name_list = [name0, name1]
    else: # if single operator
        name_list =[name]
    
    ## For each single operator, construct the circuit
    ## and append to the previous circuit if necessary
    if if_composite:
        config0 = base_circuit_code_single(name_list[0], t_list[0], basis_dict, left=left)
        config1 = base_circuit_code_single(name_list[1], t_list[1], basis_dict, left=left)
        return config0, config1
    else:
        config0 = base_circuit_code_single(name_list[0], t_list[0], basis_dict, left=left)
        return config0


def compute_right_R(name_right, ts_right, basis_dict, 
            num_spin_orb, num_particles, is_left=False, debug=False):
    """
    # For diagonal entry in H, compute R_i. e.g., 
    for computing <+R1|-R2> and <+R1|H|-R2> where |+R1> = e^{+R_1} |\psi> and 
    |-R2> = e^{- R_2} |\psi>, we have <+R1|R2+> = <\psi|e^{-R_1}e^{+R_2}|\psi> and 
    <R1|H|R2> = <\psi|e^{- R_1}He^{+R_2}|\psi>, then this function is for calculating and returning
    the matrices e^{-R_1}e^{+R_2}| and e^{-R_1}He^{+R_2}|.

    In the above example,
    Ham_mat is H
    name_left is 'R1'
    name_right is 'R2'
    ts_left is the value of t_1
    ts_right is the value of t_2
    basis_dict is the dict used to show e.g., '+R1' correspinds to matrix e^{+R_1}
    num_spin_orb is numebr of spin orbitals, for JW mapping
    num_particles is (number of alpha electron, number of beta electron), for JW mapping

    For composite, e.g., <+R2-R1|+R3+R5>, we need <\psi| e^{+R1}e^{-R2} e^{+R3}e^{+R5} |\psi>
    for handling <+R2-R1|, base_circuit_code_separate() will do the transfer e^{+R2}e^{-R1} -> e^{-R1}e^{+R2}
    and base_circuit_code_single() called inside base_circuit_code_separate() will do e^{-R1}e^{+R2} -> e^{+R1}e^{-R2}
    then e^{+R1} and e^{-R2} are returned back separately and do the dot product e^{+R1}e^{-R2}

    NOTE: do NOT put sign before R0, e.g., write 'R0' instead of '+R0' or '-R0'

    Args:
        name_right (str):basis name on the right part of the expectation, e.g. '-R2' in <+R1|H|-R2>
        ts_right (tuple, (float,) or (float, float)): the value of t for single base or t's for composite base in right part, e.g., if name_left='+R1', then ts_left=(t1,); if name_left='+R1-R2', then ts_left=(t1,t2)
        basis_dict (dict): configuration for basis in convention in Qiskit. e.g. the output of function of_to_qis_config()
        num_spin_orb (int): numebr of spin orbitals, for JW mapping
        num_particles (tuple, (int, int)): (number of alpha electron, number of beta electron), for JW mapping
        debug (bool, optional): If debug=True, then return matrices separately instead of the product, e.g., for <+R2-R1|H|+R3+R5>, return e^{+R1}e^{-R2} and H and e^{+R3}e^{+R5} . Defaults to False.

    Raises:
        Exception: do NOT put sign before R0 in name_right

    Returns:
        qiskit.opflow.primitive_ops.pauli_sum_op.PauliSumOp: product result if debug=False by default
    """
    if 'R0' in name_right and len(name_right) > 2:
        raise Exception('Do not include R0 in composite or with a sign. Not implemented.')
        
    qubit_converter = QubitConverter(mapper=JordanWignerMapper())
    ## Note: that the name order is flipped in base_circuit_code_separate()
    ## For example, in <+R2-R1| = <\psi|e^{+R1}e^{-R2}, left_sub0 is e^{+R1} and left_sub1 is e^{-R2}
    ## The name order flip is done in base_circuit_code_separate() directly
    ## the sign of t's is flipped in base_circuit_code_single() called by base_circuit_code_separate()
    if_right_composite = (name_right.count('R') > 1) # At most 2
    if if_right_composite:
        right_sub0,right_sub1 = base_circuit_code_separate(name_right, ts_right, basis_dict, left=is_left)
        right_fem_op0 = FermionicOp(right_sub0, register_length=num_spin_orb, display_format='dense')
        right_qis_op0 = qubit_converter.convert(right_fem_op0, num_particles=num_particles)

        right_fem_op1 = FermionicOp(right_sub1, register_length=num_spin_orb, display_format='dense')
        right_qis_op1 = qubit_converter.convert(right_fem_op1, num_particles=num_particles)
        return right_qis_op0, right_qis_op1
    else:
        right_sub = base_circuit_code_separate(name_right, ts_right, basis_dict, left=is_left)
        right_fem_op = FermionicOp(right_sub, register_length=num_spin_orb, display_format='dense')
        right_qis_op = qubit_converter.convert(right_fem_op, num_particles=num_particles)
    return right_qis_op



def compute_left_R(name_left, ts_left, basis_dict, 
            num_spin_orb, num_particles, debug=False): 
    # For diagonal entry in H, compute (e^{R_i})^\dagger
    ## Note: that the name order is flipped in base_circuit_code_separate()
    ## For example, in <+R2-R1| = <\psi|e^{+R1}e^{-R2}, left_sub0 is e^{+R1} and left_sub1 is e^{-R2}
    ## The name order flip is done in base_circuit_code_separate() directly
    ## the sign of t's is flipped in base_circuit_code_single() called by base_circuit_code_separate()
    
    if_left_composite = name_left.count('R') - 1
    if if_left_composite: 
        left_qis_op0, left_qis_op1 = compute_right_R(name_left, ts_left, basis_dict,
                                            num_spin_orb, num_particles, is_left=True, debug=debug)
        return left_qis_op0, left_qis_op1
    else:
        left_qis_op0 = compute_right_R(name_left, ts_left, basis_dict,
                                num_spin_orb, num_particles, is_left=True, debug=debug)
        return left_qis_op0



def compute_right_expop(name_right, ts_right, basis_dict, 
            num_spin_orb, num_particles, debug=False): # For diagonal entry in H, compute e^{R_i}
    if_right_composite = (name_right.count('R') > 1) # At most 2
    if if_right_composite:
        right_qis_op0,right_qis_op1 = compute_right_R(name_right, ts_right, basis_dict, 
                                            num_spin_orb, num_particles, is_left=False, debug=debug)
        right_qis_op = (1j*right_qis_op0).exp_i() @ (1j*right_qis_op1).exp_i()
    else:
        right_qis_op0 = compute_right_R(name_right, ts_right, basis_dict, 
                                num_spin_orb, num_particles, is_left=False, debug=debug)
        right_qis_op = (1j*right_qis_op0).exp_i() 
    return right_qis_op



def compute_left_expop(name_left, ts_left, basis_dict,
            num_spin_orb, num_particles, debug=False): 
    if_left_composite = name_left.count('R') - 1
    if if_left_composite: 
        left_qis_op0,left_qis_op1 = compute_left_R(name_left, ts_left, basis_dict, 
                                            num_spin_orb, num_particles, debug=debug)
        left_qis_op = (1j*left_qis_op0).exp_i() @ (1j*left_qis_op1).exp_i()
    else:
        left_qis_op0 = compute_left_R(name_left, ts_left, basis_dict, 
                                num_spin_orb, num_particles, debug=debug)
        left_qis_op = (1j*left_qis_op0).exp_i() 
    return left_qis_op  

def est_SH_class(Ham_mat,
                basis_set, ts, 
                basis_config, 
                init_state, 
                num_spin_orbitals, num_particles, print_progress=True):
    """
    Compute S and H. E.g., <R_i|R_j> and <R_i|H|R_j> for all R_i,R_j \in \{R0, +R1, -R1, +R2, -R2\},
    following the provided order of basis, each <R_i|R_j> and <R_i|H|R_j> will be the entry value of two matrices S and H, respectively.

    NOTE: do NOT put sign before R0, e.g., write 'R0' instead of '+R0' or '-R0'

    Args:
        Ham_mat (qiskit.opflow.primitive_ops.pauli_sum_op.PauliSumOp): qiskit operator that describes the Hamiltonian matrix
        ts_left (tuple, (float,) or (float, float)): the value of t for single base or t's for composite base in left part, e.g., if name_left='+R1', then ts_left=(t1,); if name_left='+R1-R2', then ts_left=(t1,t2)
        ts_right (tuple, (float,) or (float, float)): the value of t for single base or t's for composite base in right part, e.g., if name_left='+R1', then ts_left=(t1,); if name_left='+R1-R2', then ts_left=(t1,t2)
        basis_set (array[str]): basis of interest, e,g., ['R0','+R1','-R1','+R2','-R2','+R3','-R3','+R4','-R4','+R5','+R6','+R3+R5','+R4+R6','+R2+R1','+R2-R1','-R2+R1','-R2-R1']
        ts (array[float]): the values of t's for basis, the order must be [R0, R1, R2,...], the t value for R0 usually is 1. E.g., [1.        , 0.98341232, 8.7119191 , 4.08680637, 9.37143595,0.35234131, 5.48091233] for R0 to R6
        basis_config (dict): configuration for basis in convention in Qiskit. e.g. the output of function of_to_qis_config()
        init_state (qiskit.circuit): usually Hatree Fock state, can be constructed using HartreeFock() in Qiskit Nature
        num_spin_orbitals (int): numebr of spin orbitals, for JW mapping
        num_particles (tuple, (int, int)): (number of alpha electron, number of beta electron), for JW mapping

    Returns:
        (numpy.matrix, numpy.matrix): matrix S and H
    """

    mat_len = len(basis_set)
    mat_size = mat_len*mat_len
    S_mat = np.zeros((mat_len, mat_len), dtype=complex) # initialization
    H_mat = np.zeros((mat_len, mat_len), dtype=complex) # initialization
    for i in range(mat_len):
        nal = basis_set[i] # name left
        ## Select approriate t, first decide if single operator or composite
        if nal.count('R') == 1:
            tl = (ts[int(nal.split('R')[-1])],)
        elif nal.count('R') == 2: # composite
            nal_nopm = re.sub(r'[+-]','',nal) # remove +- signs
            nal_indlist = list(filter(None, nal_nopm.split('R'))) # remove empty strings from split
            tl = (ts[int(nal_indlist[1])], ts[int(nal_indlist[0])])
        else:
            raise ValueError('Invalid basis name for basis: {}'.format(nal))
        ## Compute exp(tR) for LHS
        if nal != 'R0':
            left_er = compute_left_expop(nal, tl, basis_config,
                                num_spin_orbitals, num_particles, debug=False)
        ## Compute exp(tR) for RHS
        for j in range(mat_len):
            if print_progress:
                print("Starting {:.2f}%".format(100*(1+i*mat_len + j)/mat_size), end='\r')
            
            nar = basis_set[j] # name right
            ## Select approriate t, first decide if single operator or composite
            if nar.count('R') == 1:
                tr = (ts[int(nar.split('R')[-1])],)
            elif nar.count('R') == 2: # composite
                nar_nopm = re.sub(r'[+-]','',nar) # remove +- signs
                nar_indlist = list(filter(None, nar_nopm.split('R'))) # remove empty strings from split
                tr = (ts[int(nar_indlist[0])], ts[int(nar_indlist[1])])
            else:
                raise ValueError('Invalid basis name for basis: {}'.format(nar))

            # NOTE: Note that for basis on the left, e.g., <+R1-R2| in <+R1-R2|H|+R5-R5>, t's are reversed
            # here, so tl=(t2,t1) in the above example, BUT NOT the BASIS NAME, the name is still '+R1-R2'
            # the reverse of the name is in base_circuit_code_separate(), 
            # and adding - sign to tl is in base_circuit_code_single()
            if nal == nar: # Diagonal Entries
                ## Compute S matrix Entry
                S_mat[i,j] = 1
                ## Compute H matrix Entry
                h_mat_measurement = StateFn(Ham_mat).adjoint()
                if nal == 'R0':
                    h_mat_measurable_expression = h_mat_measurement  @ CircuitStateFn(init_state)
                else: # other diagonal entry
                    right_er = compute_right_expop(nar, tr, basis_config, num_spin_orbitals, num_particles)
                    h_mat_measurable_expression = h_mat_measurement @ right_er @ CircuitStateFn(init_state)
                h_mat_expectation = MatrixExpectation().convert(h_mat_measurable_expression)
                H_mat[i,j] = h_mat_expectation .eval()
            else:
                if nar == 'R0':
                    s_obs = left_er
                    h_obs = left_er @ Ham_mat
                elif nal == 'R0':
                    right_er = compute_right_expop(nar, tr, basis_config,
                                        num_spin_orbitals, num_particles, debug=False)
                    s_obs = right_er
                    h_obs = Ham_mat @ right_er
                else:
                    right_er = compute_right_expop(nar, tr, basis_config,
                                        num_spin_orbitals, num_particles, debug=False)
                    s_obs = left_er @ right_er
                    h_obs = left_er @ Ham_mat @ right_er
                ## Compute S matrix Entry
                s_mat_measurement = StateFn(s_obs).adjoint()
                s_mat_measurable_expression = s_mat_measurement  @ CircuitStateFn(init_state)
                s_mat_expectation = MatrixExpectation().convert(s_mat_measurable_expression)
                S_mat[i,j] = s_mat_expectation .eval()
                ## Compute H matrix Entry
                h_mat_measurement = StateFn(h_obs).adjoint()
                h_mat_measurable_expression = h_mat_measurement  @ CircuitStateFn(init_state)
                h_mat_expectation = MatrixExpectation().convert(h_mat_measurable_expression)
                H_mat[i,j] = h_mat_expectation .eval()

    S_npmat = np.matrix(S_mat).real
    H_npmat = np.matrix(H_mat).real
    return S_npmat, H_npmat








##------------------- Use Taylor Expansion and Trotterization for exp(iA)Hexp(-iB) -------------------##

def opI(n): # identity operator
    op = I
    for _ in range(n - 1):
        op = op^I
    return op

def sep_pauli_coeff(sum_op): # Read Pauli strings and coeffs from a PauliSumOp object
    multplier = sum_op.coeff
    coeffs = []
    pstrs = []
    origin_coeffs = sum_op.coeffs
    origin_paulis = sum_op.primitive.paulis
    for i in range(len(sum_op.coeffs)):
        coeffs.append(origin_coeffs[i] * multplier)
        pstrs.append(PauliOp(origin_paulis[i]))
    return coeffs, pstrs


def expn_convert(coef, power_op):
    # for exp(itP) and P = IIXZ, t is coef, P is a power_op and must be a single Pauli term
    # P is qiskit.opflow.PauliOp
    # t is int or float
    # expand to cos(coef)Id + isin(coef)P
    id_op = opI(power_op.num_qubits)
    res_op = (np.cos(coef)*id_op) + (1j*np.sin(coef)*power_op)
    return res_op

def expn_trotter(power_opm, steps, atol = 1e-8, order = 2):
    # Find trottered exp(iA) where A = \sum_{i = 1}^m s_i P_i
    # A is qiskit.opflow.PauliSumOp
    # steps is number of Trotter steps
    # Default Suzuki-trotter order is 2
    
    ## Separate coefficients and Pauli operators
    A_coefs, A_paulis = sep_pauli_coeff(power_opm)
    ## Convert each term to cos(s)I + isin(s)P form
    converted_terms = []
    const = order*steps
    for i in range(len(A_coefs)):
        single_term = expn_convert(A_coefs[i]/const, A_paulis[i])
        converted_terms.append(single_term)
    ## Trotterize
    LHSop = converted_terms[0].copy()
    for j in range(1, len(converted_terms)):
        LHSop = (LHSop @ converted_terms[j]).reduce(atol=atol)
    if order == 1:
        one_term = LHSop.copy()
    elif order == 2:
        ## Do 2nd-order trotter steps
        RHSop = converted_terms[-1].copy()
        for k in range(len(converted_terms)-2 ,-1, -1):
            RHSop = (RHSop @ converted_terms[k]).reduce(atol=atol)
        ## Multiply the resultant op by "steps" times
        one_term = (LHSop @ RHSop).reduce(atol=atol)
    else:
        raise ValueError("Order must be 1 or 2")
    sol = one_term.copy()
    for _ in range(steps-1):
        sol = (sol @ one_term).reduce(atol=atol)
    return sol


def trotter_right(name_right, ts_right, 
                  basis_dict, 
                  num_spin_orb, num_particles, 
                  trotter_steps, atol = 1e-8, order = 2,
                  debug=False): ## Trotterize the right exp of exp(iA)Hexp(-iB)
    if_right_composite = (name_right.count('R') > 1) # At most 2
    if if_right_composite:
        right_qis_op0,right_qis_op1 = compute_right_R(name_right, ts_right, basis_dict, 
                                            num_spin_orb, num_particles, is_left=False, debug=debug)
        right_er0 = expn_trotter(-1j*right_qis_op0, trotter_steps, atol=atol, order=order)
        right_er1 = expn_trotter(-1j*right_qis_op1, trotter_steps, atol=atol, order=order)
        right_er = (right_er0 @ right_er1).reduce(atol=atol)
    else:
        right_qis_op0 = compute_right_R(name_right, ts_right, basis_dict, 
                                num_spin_orb, num_particles, is_left=False, debug=debug)
        right_er = expn_trotter(-1j*right_qis_op0, trotter_steps, atol=atol, order=order)
    return right_er

def trotter_left(name_left, ts_left, 
                 basis_dict, 
                 num_spin_orb, num_particles, 
                 trotter_steps, atol = 1e-8, order = 2,
                 debug=False): ## Trotterize the left exp of exp(iA)Hexp(-iB)
    if_left_composite = (name_left.count('R') > 1) # At most 2
    if if_left_composite:
        left_qis_op0,left_qis_op1 = compute_left_R(name_left, ts_left, basis_dict, 
                                            num_spin_orb, num_particles, debug=debug)
        left_er0 = expn_trotter(-1j*left_qis_op0, trotter_steps, atol=atol, order=order)
        left_er1 = expn_trotter(-1j*left_qis_op1, trotter_steps, atol=atol, order=order)
        left_er = (left_er0 @ left_er1).reduce(atol=atol)
    else:
        left_qis_op0 = compute_left_R(name_left, ts_left, basis_dict, 
                                num_spin_orb, num_particles, debug=debug)
        left_er = expn_trotter(-1j*left_qis_op0, trotter_steps, atol=atol, order=order)
    return left_er











## -------------- Compute Diagonal with Trotter ---------------------
def est_SH_qwc_anyentry_mt(Ham_mat,
                basis_set_row, basis_set_col, ts, 
                basis_config, 
                init_state, 
                num_spin_orbitals, num_particles,
                backend,
                num_shots, opt_level, seed,
                symmetric = False,
                trotter_steps = 1, atol=1e-8, order=2, mecls = None,
                return_circuits = False, print_progress=True, print_entry=False): # mt for more trotter
    """
    Compute S and H. E.g., <R_i|R_j> and <R_i|H|R_j> for all R_i,R_j \in \{R0, +R1, -R1, +R2, -R2\},
    following the provided order of basis, each <R_i|R_j> and <R_i|H|R_j> will be the entry value of two matrices S and H, respectively.

    NOTE: do NOT put sign before R0, e.g., write 'R0' instead of '+R0' or '-R0'

    Args:
        Ham_mat (qiskit.opflow.primitive_ops.pauli_sum_op.PauliSumOp): qiskit operator that describes the Hamiltonian matrix
        ts_left (tuple, (float,) or (float, float)): the value of t for single base or t's for composite base in left part, e.g., if name_left='+R1', then ts_left=(t1,); if name_left='+R1-R2', then ts_left=(t1,t2)
        ts_right (tuple, (float,) or (float, float)): the value of t for single base or t's for composite base in right part, e.g., if name_left='+R1', then ts_left=(t1,); if name_left='+R1-R2', then ts_left=(t1,t2)
        basis_set (array[str]): basis of interest, e,g., ['R0','+R1','-R1','+R2','-R2','+R3','-R3','+R4','-R4','+R5','+R6','+R3+R5','+R4+R6','+R2+R1','+R2-R1','-R2+R1','-R2-R1']
        ts (array[float]): the values of t's for basis, the order must be [R0, R1, R2,...], the t value for R0 usually is 1. E.g., [1.        , 0.98341232, 8.7119191 , 4.08680637, 9.37143595,0.35234131, 5.48091233] for R0 to R6
        basis_config (dict): configuration for basis in convention in Qiskit. e.g. the output of function of_to_qis_config()
        init_state (qiskit.circuit): usually Hatree Fock state, can be constructed using HartreeFock() in Qiskit Nature
        num_spin_orbitals (int): numebr of spin orbitals, for JW mapping
        num_particles (tuple, (int, int)): (number of alpha electron, number of beta electron), for JW mapping
        backend (qiskit.providers.aer.backends.aer_simulator.AerSimulator): the backend to run the quantum circuit, can be the real backend or simulator
        num_shots (int/long): number of shots for each evaluation of a quantum circuit
        opt_level (optimization level of transpiler): Qiskit can optimize the circuit classically, 0 is no optimization, 3 is the highest optimization, usually is 1
        seed (int): the random seed for simulator and transipler (transpile also has some randomness)
        trotter_steps (int): number of Trotter steps
        symmetric (bool): if True, the S and H are symmetric, otherwise, S and H are not symmetric
        return_circuits (bool): if True, return the circuit for each entry of S and H

    Returns:
        (numpy.matrix, numpy.matrix): matrix S and H
    """

    circ_dict = {}
    pauli_dict = {}
    mat_len_row = len(basis_set_row)
    mat_len_col = len(basis_set_col)
    if mat_len_row != mat_len_col:
        symmetric = False
    mat_size = mat_len_row*mat_len_col
    counter = 0
    if symmetric:
        counting_size = int(mat_len_row*(mat_len_col+1)/2)
    else:
        counting_size = int(mat_len_row*mat_len_col)
    S_mat = np.zeros((mat_len_row, mat_len_col), dtype=complex) # initialization
    H_mat = np.zeros((mat_len_row, mat_len_col), dtype=complex) # initialization
    for i in range(mat_len_row):
        nal = basis_set_row[i] # name left
        ## Select approriate t, first decide if single operator or composite
        if nal.count('R') == 1:
            tl = (ts[int(nal.split('R')[-1])],)
        elif nal.count('R') == 2: # composite
            nal_nopm = re.sub(r'[+-]','',nal) # remove +- signs
            nal_indlist = list(filter(None, nal_nopm.split('R'))) # remove empty strings from split
            tl = (ts[int(nal_indlist[1])], ts[int(nal_indlist[0])])
        else:
            raise ValueError('Invalid basis name for basis: {}'.format(nal))
        if nal != 'R0':
            left_er = trotter_left(nal, tl, basis_config,
                                num_spin_orbitals, num_particles, trotter_steps, atol=atol, order=order,
                                debug=False)
            left_half_h_obs = (left_er @ Ham_mat).reduce(atol=atol)
        if symmetric:
            j_range = range(i, mat_len_col)
        else:
            j_range = range(mat_len_col)
        for j in j_range:
            counter += 1
            q_ins = QuantumInstance(backend=backend, 
                                shots=int(num_shots), 
                                seed_simulator=seed,
                                seed_transpiler=seed,
                                optimization_level=opt_level,
                                noise_model=None,
                                measurement_error_mitigation_cls=mecls, cals_matrix_refresh_period=1440)

            if print_progress:
                print("Starting {:.2f}%".format(100*(counter)/counting_size), end='\r')

            nar = basis_set_col[j] # name right
            if return_circuits:
                circ_dict[nal+'_'+nar] = {'S':[], 'H': []}
                pauli_dict[nal+'_'+nar] = {'S':0, 'H': 0}
            ## Select approriate t, first decide if single operator or composite
            if nar.count('R') == 1:
                tr = (ts[int(nar.split('R')[-1])],)
            elif nar.count('R') == 2: # composite
                nar_nopm = re.sub(r'[+-]','',nar) # remove +- signs
                nar_indlist = list(filter(None, nar_nopm.split('R'))) # remove empty strings from split
                tr = (ts[int(nar_indlist[0])], ts[int(nar_indlist[1])])
            else:
                raise ValueError('Invalid basis name for basis: {}'.format(nar))

            # NOTE: Note that for basis on the left, e.g., <+R1-R2| in <+R1-R2|H|+R5-R5>, t's are reversed
            # here, so tl=(t2,t1) in the above example, BUT NOT the BASIS NAME, the name is still '+R1-R2'
            # the reverse of the name is in base_circuit_code_separate(), 
            # and adding - sign to tl is in base_circuit_code_single()
            if nal == nar and nal == 'R0': # Diagonal Entries
                ## Compute S matrix Entry
                S_mat[i,j] = 1
                ## Compute H matrix Entry
                sampler_h = CircuitSampler(q_ins)
                h_mat_measurement = StateFn(Ham_mat).adjoint()
                h_mat_measurable_expression = h_mat_measurement  @ CircuitStateFn(init_state)
                h_mat_trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(reps=1, order=order)).convert(h_mat_measurable_expression)
                h_mat_expectation = PauliExpectation(group_paulis=True).convert(h_mat_trotterized_op)
                h_mat_sampled_exp_op = sampler_h.convert(h_mat_expectation)
                H_mat[i,j] = h_mat_sampled_exp_op.eval()
                if print_entry:
                    print(nal,',', nar,':', S_mat[i,j],',', H_mat[i,j])
                # Save the circuit
                if return_circuits:
                    pauli_dict[nal+'_'+nar]['H'] = len(Ham_mat)
                    h_circ_ops = list(sampler_h._circuit_ops_cache.values())
                    for hc in h_circ_ops:
                        circ_dict[nal+'_'+nar]['H'].append(hc.to_circuit())
            else:
                if nar == 'R0':
                    s_obs = left_er
                    h_obs = left_half_h_obs
                elif nal == 'R0':
                    right_er = trotter_right(nar, tr, basis_config,
                                    num_spin_orbitals, num_particles, trotter_steps, atol=atol, order=order,
                                    debug=False)  
                    s_obs = right_er
                    h_obs = (Ham_mat @ right_er).reduce(atol=atol)
                else:
                    right_er = trotter_right(nar, tr, basis_config,
                                    num_spin_orbitals, num_particles, trotter_steps, atol=atol, order=order,
                                    debug=False)  
                    s_obs = (left_er @ right_er).reduce(atol=atol)
                    h_obs = (left_half_h_obs @ right_er).reduce(atol=atol)
                ## Compute S matrix Entry
                sampler_s = CircuitSampler(q_ins)
                s_mat_measurement = StateFn(s_obs).adjoint()
                s_mat_measurable_expression = s_mat_measurement  @ CircuitStateFn(init_state)
                s_mat_expectation = PauliExpectation(group_paulis=True).convert(s_mat_measurable_expression)
                s_mat_sampled_exp_op = sampler_s.convert(s_mat_expectation)
                S_mat[i,j] = s_mat_sampled_exp_op.eval()
                if symmetric:
                    S_mat[j,i] = S_mat[i,j].conjugate()
                ## Compute H matrix Entry
                sampler_h = CircuitSampler(q_ins)
                h_mat_measurement = StateFn(h_obs).adjoint()
                h_mat_measurable_expression = h_mat_measurement  @ CircuitStateFn(init_state)
                h_mat_expectation = PauliExpectation(group_paulis=True).convert(h_mat_measurable_expression)
                h_mat_sampled_exp_op = sampler_h.convert(h_mat_expectation)
                H_mat[i,j] = h_mat_sampled_exp_op.eval() # double
                if symmetric:
                    H_mat[j,i] = H_mat[i,j].conjugate()
                if print_entry:
                    print(nal,',', nar,':', S_mat[i,j],',', H_mat[i,j])
                # ## Save the circuit
                if return_circuits:
                    pauli_dict[nal+'_'+nar]['S'] = len(s_obs)
                    pauli_dict[nal+'_'+nar]['H'] = len(h_obs)
                    s_circ_ops = list(sampler_s._circuit_ops_cache.values())
                    h_circ_ops = list(sampler_h._circuit_ops_cache.values())
                    for sc in s_circ_ops:
                        circ_dict[nal+'_'+nar]['S'].append(sc.to_circuit())
                    for hc in h_circ_ops:
                        circ_dict[nal+'_'+nar]['H'].append(hc.to_circuit())
    
    S_npmat = np.matrix(S_mat)
    H_npmat = np.matrix(H_mat)
    if return_circuits:
        return S_npmat, H_npmat, circ_dict, pauli_dict
    return S_npmat, H_npmat


def est_SH_qwc(Ham_mat,
            basis_set, ts, 
            basis_config, 
            init_state, 
            num_spin_orbitals, num_particles,
            backend,
            num_shots, opt_level, seed,
            symmetric = False,
            trotter_steps = 1, atol=1e-8, order=2, mecls = None,
            return_circuits = False, print_progress = True, print_entry=False):
    #  when basis_set_row = basis_set_col = basis_set
    return est_SH_qwc_anyentry_mt(Ham_mat,
                        basis_set, basis_set, ts, 
                        basis_config, 
                        init_state, 
                        num_spin_orbitals, num_particles,
                        backend,
                        num_shots, opt_level, seed,
                        symmetric = symmetric,
                        trotter_steps = trotter_steps, atol=atol, order=order, mecls = mecls,
                        return_circuits = return_circuits,
                        print_progress = print_progress, print_entry=print_entry)
#------------------------------------------------------------------------------------#




def plot_sh(S, H, basis_displayed, save_name=None, exp_name=''):
    """
    Helper function to plot S and H side-by-side
    Example plot_sh(S, H, ['R0', '+R1', '-R1'], save_name='./Data/SH.pdf', exp_name='{R0,R1} experiment')
    """
    S_title_str = '{:s}, S'.format(exp_name) if exp_name else 'S'
    H_title_str = '{:s}, H'.format(exp_name) if exp_name else 'H'
    
    figure(figsize=(12, 6), dpi=100)
    # S
    plt.subplot(1, 2, 1)
    plt.imshow(S)
    plt.colorbar()
    plt.xticks(list(range(len(basis_displayed))), basis_displayed, rotation = 90)
    plt.yticks(list(range(len(basis_displayed))), basis_displayed)
    plt.title(S_title_str)
    # plt.clim(color_lb, color_ub) 
    plt.tight_layout()
    # H
    plt.subplot(1, 2, 2)
    plt.imshow(H)
    plt.colorbar()
    plt.xticks(list(range(len(basis_displayed))), basis_displayed, rotation = 90)
    plt.yticks(list(range(len(basis_displayed))), basis_displayed)
    plt.title(H_title_str)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    plt.show()
































