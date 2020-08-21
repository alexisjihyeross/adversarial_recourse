from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *

weights_to_eval = [0.0, 0.05, 0.1, 0.25, 0.5]
thresholds_to_eval = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

bail_experiment_dir = 'new_results/0815_bail/'
bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names = process_bail_data()
bail_data = read_data(bail_experiment_dir)

delta_max = 0.75

data = bail_data
experiment_dir = bail_experiment_dir
actionable_indices = bail_actionable_indices

for w in weights_to_eval:
    print("WEIGHT: ", w)    
    weight_dir = experiment_dir + str(w) + "/"
    model = load_torch_model(weight_dir, w)

	# Runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function
    # run_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, lam_init = 0.005, data_indices = range(0, 250), thresholds = thresholds_to_eval)

    epsilons = [0.7, 0.8, 0.9, 0.95]
    d = 0.95
    data_indices = range(0, 250)
    compute_threshold_upperbounds(model, data['X_test'], data['y_test'], w, delta_max, data_indices, actionable_indices, epsilons, d, weight_dir)
