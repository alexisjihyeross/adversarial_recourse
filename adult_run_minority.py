from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *

weights_to_eval = [0.0, 0.45]
# thresholds_to_eval = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

adult_experiment_dir = 'new_results/all/0907_adult_0.75/'
adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
adult_data = read_data(adult_experiment_dir)

delta_max = 0.75

data = adult_data
experiment_dir = adult_experiment_dir
actionable_indices = adult_actionable_indices
categorical_features = adult_categorical_features
white_feature_name = "isWhite"
data_indices = range(0, 250)

weights_to_eval = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
for w in weights_to_eval:
    print("WEIGHT: ", w)    
    weight_dir = experiment_dir + str(w) + "/"
    model = load_torch_model(weight_dir, w)    
    # Runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function
    run_minority_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, white_feature_name, lam_init = 0.001, data_indices = data_indices)
