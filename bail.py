from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *

bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names = process_bail_data()
bail_experiment_dir = 'new_results/0907_bail_0.75/'

bail_data = get_data(bail_X, bail_y)
write_data(bail_data, bail_experiment_dir)

# bail_data = read_data(bail_experiment_dir)

delta_max = 0.75
thresholds_to_eval = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

run(bail_data, bail_actionable_indices, bail_categorical_features, bail_experiment_dir, [0.0, 0.05, 0.1, 0.25, 0.5], delta_max, do_train = True, thresholds_to_eval = thresholds_to_eval)
