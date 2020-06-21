"""

"""

import pickle

import attr
import numpy as np

from hdp_var.model.hdp_var import HDPVar
from hdp_var.parameters import TrainingParams, SamplingParams
from hdp_var.utils.data_preparation import generate_data_structure
from hdp_var.utils.stats import median_r_2
from hdp_var.utils.plot import plot, plot_1, plot_likelihood

data_path = '/Users/benjaminfuhrer/GitHub/hdp_var_python/data/'
file_name = 'fes10.pkl'
L = 20
order = 2
model_name = 'model_3'
to_train = False

with open(f'{data_path}{file_name}', 'rb') as f:
    data = pickle.load(f)


train_data = generate_data_structure(data['data'], order)
test_data = generate_data_structure(data['test_data'], order)
test_labels = data['test_labels'][order:]
D = train_data['Y'].shape[0]

#
r2_s = []
i = 1
while i <= 1:
    print('**********************************************************')
    print(f'model: {i}')
    if to_train:
        model = HDPVar(D, L, order)
        tr = attr.asdict(TrainingParams(iterations=500, sample_every=10, burn_in=100, print_every=10))
        s_params = attr.asdict(SamplingParams(S_0=np.eye(D), b_gamma=0.001, b_alpha=0.001))
        model.set_training_parameters(tr)
        print(model.training_parameters)
        model.set_sampling_parameters(s_params)
        print(model.sampling_parameters)
        model.train(train_data)
        with open(f'{data_path}{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(f'{data_path}{model_name}.pkl', 'rb') as f:
            model = pickle.load(f)

    state_sequence = model.predict_state_sequence(test_data)
    pred_Y = model.predict_data(X_0=test_data['X'], reset_every=100)
    r2 = median_r_2(y=test_data['Y'], pred_y=pred_Y)
    r2_s.append(r2)

    print(r2)
    print('**********************************************************')
    i += 1
    # plot_likelihood(model)
    # plot(test_data['Y'][0], pred_Y[0])
    # plot_1(test_data['Y'][0])
print(r2_s)
print('')
