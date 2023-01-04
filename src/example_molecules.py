from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from qugcm import readHam, of_to_qis_config


##------------------------------------ Experiment Parameter Data -----------------------------------------------##
QUBIT_CONVERTER = QubitConverter(mapper=JordanWignerMapper())

class ExpMole:
  def __init__(self, total_energy, repulsion_energy, HF_energy, 
               num_spin_orbitals, num_particles,
               ham_path,
               mapping):
    ## Energies
    self.total_energy = total_energy
    self.rep_energy = repulsion_energy
    self.HF_energy = HF_energy
    self.total_wcrep = self.total_energy - self.rep_energy # total energy without repulsion energy
    self.HF_wcrep = self.HF_energy - self.rep_energy # HF energy without repulsion energy
    ## Particals and Orbitals
    self.norb= num_spin_orbitals
    self.nptc = num_particles # (num alpha particles, num beta particles)
    ## Orbital mapping: TCE to whatever that Qiskit uses
    self.mapping = mapping
    ## Read Hamilonian
    try:
        self.ham_fm = readHam(ham_path, self.norb, mapping=self.mapping)
    except:
        self.ham_fm = readHam('../'+ham_path, self.norb, mapping=self.mapping)
    ## Jordan-Wigner transformed Hamilonian
    self.ham_op = QUBIT_CONVERTER.convert(self.ham_fm, num_particles=self.nptc)

## Molecule informations: energies, number of orbitals, number of alpha and beta electrons, Hamiltonian
data_folder ='./Example_Data/'  ## <-- NOTE: becareful with this data folder in case anything changes
H4_MAPPING = {
        '0': '0',
        '1': '1',
        '2': '4',
        '3': '5',
        '4': '2',
        '5': '3',
        '6': '6',
        '7': '7'}



H4L = ExpMole(-2.151003659163793, 2.1666666667, -2.0752429332999998,
             8, (2,2),
             data_folder + 'h4_linear_ham.txt',
             H4_MAPPING)# alpha = 0.500
H4S = ExpMole(-1.942993405043510, 2.6863890759,-1.791585507768,
             8, (2,2),
             data_folder + 'h4_square_ham.txt',
             H4_MAPPING) # alpha = 0.005                             


## Configuration of Basis
def single_op_fillin(a,b,c,d): # a, b ,c ,d are digits, (a^+ b - c^+ d) - (b^+ a - d^+ c)
    a = str(a)
    b = str(b)
    c = str(c)
    d = str(d)
    return [(a+'^ '+b, 1), (c+'^ '+d,-1), (b+'^ '+a,-1), (d+'^ '+c, 1)]

def single_op_fillin_alt(a,b,c,d): # a, b ,c ,d are digits, (a^+ b + c^+ d) - (b^+ a + d^+ c)
    a = str(a)
    b = str(b)
    c = str(c)
    d = str(d)
    return [(a+'^ '+b, 1), (c+'^ '+d,1), (b+'^ '+a,-1), (d+'^ '+c, -1)]

H4_DEFAULT_BASIS_CONFIG_OF = {
    'R1': single_op_fillin(5,2,7,4),
    'R2': single_op_fillin(6,1,8,3),
    'R3': single_op_fillin(6,2,8,4),
    'R4': single_op_fillin(5,1,7,3),
    'R5': single_op_fillin_alt(5,1,7,3),
    'R6': single_op_fillin_alt(6,2,8,4),
}
H4_ALT_BASIS_CONFIG_OF = {
    'R1': single_op_fillin_alt(5,2,7,4),
    'R2': single_op_fillin_alt(6,1,8,3),
    'R3': single_op_fillin_alt(6,2,8,4),
    'R4': single_op_fillin_alt(5,1,7,3),
    'R5': single_op_fillin_alt(5,1,7,3),
    'R6': single_op_fillin_alt(6,2,8,4),
}
H4_DEFAULT_BASIS_CONFIG_QIS = of_to_qis_config(H4_DEFAULT_BASIS_CONFIG_OF, H4_MAPPING, starting_index=1) 
H4_ALT_BASIS_CONFIG_QIS = of_to_qis_config(H4_ALT_BASIS_CONFIG_OF, H4_MAPPING, starting_index=1) 




# ## Create Hatree-Fock state
# ## option 1
# from qiskit import QuantumCircuit
# init_state = QuantumCircuit(num_spin_orbitals, name='HF')
# for i in range(len(init_state_str[::-1])):
#     s = init_state_str[::-1][i]
#     if int(s):
#         init_state.x(i)
# 
# # # option 2 foir creating initial state
# from qiskit_nature.circuit.library import HartreeFock
# # Use Hartree Fock state and add it to the main circuit
# init_state = HartreeFock(num_spin_orbitals=num_spin_orbitals, 
#                          num_particles=num_particles,
#                          qubit_converter=qubit_converter)