from utils.data_utils import *
from utils.train_utils import *
from utils.other_utils import *

delta_max = 0.75

adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names = process_adult_data()
adult_experiment_dir = 'results/adult_' + str(delta_max) + '/'

adult_data = get_data(adult_X, adult_y)
#write_data(adult_data, adult_experiment_dir)

adult_data = read_data(adult_experiment_dir)

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#run(adult_data, adult_actionable_indices, adult_categorical_features, adult_experiment_dir, weights, delta_max, do_train = True)

data = adult_data
experiment_dir = adult_experiment_dir
actionable_indices = adult_actionable_indices
categorical_features = adult_categorical_features
white_feature_name = "isWhite"
data_indices = range(0, 500)

for w in weights:
    print("minority WEIGHT: ", w)    
    weight_dir = experiment_dir + str(w) + "/"
    model = load_torch_model(weight_dir, w)    
    # Runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function
    run_minority_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, white_feature_name, lam_init = 0.001, data_indices = data_indices)
"""
for w in weights:
    print("lime WEIGHT: ", w)    
    weight_dir = experiment_dir + str(w) + "/"
    model = load_torch_model(weight_dir, w)   

    threshold_df = get_threshold_info(weight_dir, w)
    thresholds = list(threshold_df['thresholds'])

    f1s = threshold_df['f1s'] 

    # only evaluate at the threshold that maximizes f1 score on val data
    eval_thresholds = [thresholds[np.argmax(f1s)]]

    threshold = eval_thresholds[0]
    assert len(eval_thresholds) == 1    

    #for kernel_width in [0.5, 1, 1.5, 2.0]:
    for kernel_width in [0.5]:
        lime_berk_evaluate(model, data['X_train'], data['X_test'], data['y_test'], w, threshold, data_indices, actionable_indices, categorical_features, weight_dir, kernel_width)
"""
