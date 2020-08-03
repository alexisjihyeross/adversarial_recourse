from data_utils import *
from train_utils import *
from model import *
from utils import *

bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names = process_bail_data()
bail_experiment_dir = 'results/0802_bail/'
bail_data = read_data(bail_experiment_dir)
run(bail_data, bail_actionable_indices, bail_experiment_dir, [0.1, 0.25, 0.35, 0.5, 0.75, 1.0])
