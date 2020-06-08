"""

"""

import pickle
import attr
from hdp_var.utils.data_preparation import generate_data_structure
from hdp_var.model.hdp_var import HDPVar
from hdp_var.utils.HMM import viterbi
from hdp_var.parameters import TrainingParams

data_path = '/Users/benjaminfuhrer/GitHub/hdp_var_python/data/'
file_name = 'luke_data.pkl'

with open(f'{data_path}{file_name}', 'rb') as f:
    data = pickle.load(f)

L = 20
order = 2

data = generate_data_structure(data['data'], order)
D = data['Y'].shape[0]

model = HDPVar(D, L, order)
tr = attr.asdict(TrainingParams(iterations=500, sample_every=25, burn_in=20))
model.set_training_parameters(tr)
print(model.training_parameters)

state_sequence = model.train(data)
print('')
