from data_utils import *
from train_utils import *
from model import *
from utils import *


def get_threshold_info(model_dir, weight):
    # this is based on the value of this name in the train function
    best_model_thresholds_file_name = model_dir + str(weight) + '_val_thresholds_info.csv'
    threshold_df = pd.read_csv(best_model_thresholds_file_name, dtype=np.float64, index_col = 'index')
    return threshold_df


def run_wachter(model, data, w, delta_max, actionable_indices, experiment_dir, \
    thresholds = None, lam_init = 0.01, data_indices = range(0, 10)):
    """
    Runs wachter evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function

    :param model: pytorch model to evaluate
    :param data: data directionary
    :param w: weight
    :param delta_max
    :param actionable_indices
    :param experiment_dir: directory of experiment. holds all training and eval info for that weight
    :param thresholds: OPTIONAL parameter specifying what thresholds to consider
        if not supplied, read thresholds from dataframe output during training/validation
    :param lam_init: hyperparameter (passed to wachter_evaluate)
    :param data_indices: indices of data subset to evaluate on (passed to wachter_evaluate)

    """



    # define the data indices to consider
    model_dir = experiment_dir + str(w) + "/"

    # name of file where to output all results for different thresholds
    wachter_thresholds_file_name = model_dir + "wachter/" + "thresholds_test_results.csv"

    # lists in which to store results for diff thresholds
    wachter_thresholds, wachter_precisions, wachter_flipped_proportion, wachter_recourse_proportion = [], [], [], []

    # thresholds arg not supplied, read in thresholds from validation evaluation during training and use those
    if thresholds == None:
        threshold_df = get_threshold_info(model_dir, w)
        thresholds = list(threshold_df['thresholds'])

    for threshold in thresholds:
        flipped_proportion, precision, recourse_fraction = wachter_evaluate(model, data['X_test'], data['y_test'], w, threshold, delta_max, lam_init, data_indices, actionable_indices, model_dir)
        wachter_thresholds.append(threshold)
        wachter_precisions.append(precision)
        wachter_flipped_proportion.append(flipped_proportion)
        wachter_recourse_proportion.append(recourse_fraction)

    wachter_thresholds_data = {}
    wachter_thresholds_data['thresholds'] = wachter_thresholds
    wachter_thresholds_data['precisions'] = wachter_precisions
    wachter_thresholds_data['flipped_proportion'] = wachter_flipped_proportion
    wachter_thresholds_data['recourse_proportion'] = wachter_recourse_proportion

    wachter_thresholds_df = pd.DataFrame(data=wachter_thresholds_data)
    wachter_thresholds_df['thresholds'] = wachter_thresholds_df['thresholds'].round(3)
    wachter_thresholds_df.to_csv(wachter_thresholds_file_name, index_label='index')

def run(data, actionable_indices, experiment_dir, weights):
    
    lr = 0.002
    delta_max = 0.75
    fixed_precisions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for w in weights:
        print("WEIGHT: ", w)
        model = Model(len(data['X_train'].values[0]))
        
        torch_X_train = torch.from_numpy(data['X_train'].values[0:100]).float()
        torch_y_train = torch.from_numpy(data['y_train'].values[0:100]).float()
        torch_X_val = torch.from_numpy(data['X_val'].values).float()
        torch_y_val = torch.from_numpy(data['y_val'].values).float()
        
        # train the model
        train(model, torch_X_train, torch_y_train, \
             torch_X_val, torch_y_val, actionable_indices, experiment_dir, \
              recourse_loss_weight = w, num_epochs = 2, delta_max = delta_max, lr=lr, \
              fixed_precisions = fixed_precisions)


        run_wachter(model, data, w, delta_max, actionable_indices, experiment_dir, lam_init = 0.01)


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
