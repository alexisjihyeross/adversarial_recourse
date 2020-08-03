from data_utils import *
from train_utils import *
from model import *
from utils import *

def run(data, actionable_indices, experiment_dir, weights):
    
    lr = 0.002
    delta_max = 0.75
    fixed_precisions = [0.4, 0.5, 0.6, 0.7]

    for w in weights:
        print("WEIGHT: ", w)
        model = Model(len(data['X_train'].values[0]))
        
        torch_X_train = torch.from_numpy(data['X_train'].values).float()
        torch_y_train = torch.from_numpy(data['y_train'].values).float()
        torch_X_val = torch.from_numpy(data['X_val'].values).float()
        torch_y_val = torch.from_numpy(data['y_val'].values).float()
        
        # train the model
        train(model, torch_X_train, torch_y_train, \
             torch_X_val, torch_y_val, actionable_indices, experiment_dir, \
              recourse_loss_weight = w, num_epochs = 3, delta_max = delta_max, lr=lr, \
              fixed_precisions = fixed_precisions)


        run_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, lam_init = 0.01)


# adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
# adult_data = get_data(adult_X, adult_y)
# adult_experiment_dir = 'results/0802_adult/'
# write_data(adult_data, adult_experiment_dir)
# run(adult_data, adult_actionable_indices, adult_experiment_dir, [0.0, 0.015, 0.025, 0.05, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0])

bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names = process_bail_data()
bail_data = get_data(bail_X, bail_y)
bail_experiment_dir = 'results/0802_bail/'
write_data(bail_data, bail_experiment_dir)
run(bail_data, bail_actionable_indices, bail_experiment_dir, [0.0, 0.015, 0.025, 0.3, 0.05, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0])
