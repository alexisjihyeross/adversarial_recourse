from data_utils import *
from train_utils import *
from model import *
from utils import *

adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
adult_data = get_data(adult_X, adult_y)

bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names = process_bail_data()
bail_data = get_data(bail_X, bail_y)


data = adult_data
actionable_indices = adult_actionable_indices
output_dir = 'results/0802adult/' # make sure this ends with a slash

def run(data, actionable_indices, output_dir):
    
    weights = [0.0, 0.015, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    lr = 0.002
    delta_max = 0.5

    for w in weights:
        print("WEIGHT: ", w)
        model = Model(len(data['X_train'].values[0]))
        
        torch_X_train = torch.from_numpy(data['X_train'].values).float()
        torch_y_train = torch.from_numpy(data['y_train'].values).float()
        torch_X_val = torch.from_numpy(data['X_val'].values).float()
        torch_y_val = torch.from_numpy(data['y_val'].values).float()
        
        # train the model
        train(model, torch_X_train, torch_y_train, \
             torch_X_val, torch_y_val, actionable_indices, output_dir, \
              recourse_loss_weight = w, num_epochs = 10, delta_max = delta_max, lr=lr)

run(adult_data, actionable_indices, output_dir)