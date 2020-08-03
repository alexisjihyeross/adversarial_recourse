from data_utils import *
from train_utils import *
from model import *
from utils import *


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
