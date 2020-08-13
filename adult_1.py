from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *


adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
adult_experiment_dir = 'results/08_12/'
adult_data = get_data(adult_X, adult_y)
write_data(adult_data, adult_experiment_dir)
adult_data = read_data(adult_experiment_dir)
run(adult_data, adult_actionable_indices, adult_experiment_dir, [0.0, 0.015, 0.025, 0.3, 0.05])
