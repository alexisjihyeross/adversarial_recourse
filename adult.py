from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *

adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
adult_experiment_dir = 'new_results/0907_adult_0.5/'
# adult_data = read_data(adult_experiment_dir)

adult_data = get_data(adult_X, adult_y)
write_data(adult_data, adult_experiment_dir)

delta_max = 0.5
thresholds_to_eval = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

run(adult_data, adult_actionable_indices, adult_categorical_features, adult_experiment_dir, [0.0, 0.05, 0.1, 0.25, 0.5], delta_max, do_train = True, thresholds_to_eval = thresholds_to_eval)
