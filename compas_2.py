from data_utils import *
from train_utils import *
from model import *
from utils import *

compas_X, compas_y, compas_actionable_indices, compas_categorical_features, compas_categorical_names = process_compas_data()
compas_experiment_dir = 'results/0802_compas/'
compas_data = read_data(compas_experiment_dir)
run(compas_data, compas_actionable_indices, compas_experiment_dir, [0.1, 0.25, 0.35, 0.5, 0.75, 1.0])
