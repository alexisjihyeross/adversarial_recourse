from data_utils import *
from train_utils import *
from small_model import *
from big_model import *
from utils import *

weights_to_eval = [0.0, 0.05, 0.1, 0.25, 0.5]
thresholds_to_eval = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

adult_experiment_dir = 'new_results/0815_adult/'
adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
adult_data = read_data(adult_experiment_dir)

delta_max = 0.75

data = adult_data
experiment_dir = adult_experiment_dir
actionable_indices = adult_actionable_indices
categorical_features = adult_categorical_features

for w in weights_to_eval:
    print("WEIGHT: ", w)    
    weight_dir = experiment_dir + str(w) + "/"
    model = load_torch_model(weight_dir, w)

    # Runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function
    # run_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, lam_init = 0.005, data_indices = range(0, 250), thresholds = thresholds_to_eval)

    epsilons = [0.7, 0.8, 0.9, 0.95]
    d = 0.95
    data_indices = range(0, 250)
    # compute_threshold_upperbounds(model, data['X_test'], data['y_test'], w, delta_max, data_indices, actionable_indices, epsilons, d, weight_dir)
    

    kernel_widths = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    for kw in kernel_widths:
        lime_thresholds, lime_precisions, lime_flipped_proportions, lime_recourse_proportions, lime_f1s, lime_recalls, lime_accs = [], [], [], [], [], [], []


        for threshold in thresholds_to_eval:
            threshold = round(threshold, 3)
            print("THR: ", threshold)

            flipped_proportion, precision, recourse_fraction, f1, recall, acc = lime_berk_evaluate(model, data['X_train'], data['X_test'], data['y_test'], w, threshold, data_indices, actionable_indices, categorical_features, weight_dir, kernel_width = kw)
            lime_thresholds.append(threshold)
            lime_precisions.append(precision)
            lime_flipped_proportions.append(flipped_proportion)
            lime_recourse_proportions.append(recourse_fraction)
            lime_f1s.append(f1)
            lime_recalls.append(recall)
            lime_accs.append(acc)

        file_name = weight_dir + "test_eval/lime_eval/" + "lime-berk-kw-" + str(kw) +  "_thresholds_test_results.csv"

        write_threshold_info(weight_dir, w, file_name, lime_thresholds, lime_f1s, lime_accs, lime_precisions, lime_recalls, lime_flipped_proportions, lime_recourse_proportions)


