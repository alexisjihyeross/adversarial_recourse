from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *

bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names = process_bail_data()
bail_data = get_data(bail_X, bail_y)
bail_experiment_dir = 'results/0815_bail_tf/'
write_data(bail_data, bail_experiment_dir)

adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
adult_data = get_data(adult_X, adult_y)
adult_experiment_dir = 'results/0815_adult_tf/'
write_data(adult_data, adult_experiment_dir)

compas_X, compas_y, compas_actionable_indices, compas_categorical_features, compas_categorical_names = process_compas_data()
compas_data = get_data(compas_X, compas_y)
compas_experiment_dir = 'results/0815_compas_tf/'
write_data(compas_data, compas_experiment_dir)
