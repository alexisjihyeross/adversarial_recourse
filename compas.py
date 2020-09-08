from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *

# compas_experiment_dir = 'new_results/0903_compas_0.5/'
compas_experiment_dir = 'new_results/0907_compas_0.75/'
compas_X, compas_y, compas_actionable_indices, compas_categorical_features, compas_categorical_names = process_compas_data()
# compas_data = read_data(compas_experiment_dir)

compas_data = get_data(compas_X, compas_y)
write_data(compas_data, compas_experiment_dir)

delta_max = 0.75
thresholds_to_eval = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

run(compas_data, compas_actionable_indices, compas_categorical_features, compas_experiment_dir, [0.0, 0.05, 0.1, 0.25, 0.5], delta_max, do_train = True, thresholds_to_eval = thresholds_to_eval)
