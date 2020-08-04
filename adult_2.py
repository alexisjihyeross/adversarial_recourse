from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *


adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
adult_experiment_dir = 'results/0802_adult/'
adult_data = read_data(adult_experiment_dir)
run(adult_data, adult_actionable_indices, adult_experiment_dir, [0.1, 0.25, 0.35, 0.5, 0.75, 1.0])
