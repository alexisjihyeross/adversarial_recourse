from utils.data_utils import *
from utils.train_utils import *
from utils.other_utils import *

def main(data, delta_max, weights, kernel_widths, epsilons, d):

    assert data in ["adult", "compas", "bail"]

    if data == "adult":
        X, y, actionable_indices, categorical_features, _ = process_adult_data()
        white_feature_name = "isWhite"
    elif data == "compas":
        X, y, actionable_indices, categorical_features, _ = process_compas_data()
        white_feature_name = "isCaucasian"
    elif data == "bail":
        X, y, actionable_indices, categorical_features, _ = process_bail_data()
        white_feature_name = "WHITE"

    experiment_dir = 'results/' + data + '_' + str(delta_max) + '/'

    data = get_data(X, y)
    write_data(data, experiment_dir)
    # data = read_data(experiment_dir)

    run(data, actionable_indices, categorical_features, experiment_dir, weights, delta_max, do_train = True)


    data_indices = range(0, 500)

    for w in weights:
        print("minority WEIGHT: ", w)    
        weight_dir = experiment_dir + str(w) + "/"
        model = load_torch_model(weight_dir, w)    

        # Runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function
        run_minority_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, white_feature_name, lam_init = 0.001, data_indices = data_indices)

        """ LIME EVALUATION """
        # only evaluate at the threshold that maximizes f1 score on val data
        threshold_df = get_threshold_info(weight_dir, w)
        thresholds = list(threshold_df['thresholds'])
        f1s = threshold_df['f1s'] 
        eval_thresholds = [thresholds[np.argmax(f1s)]]
        threshold = eval_thresholds[0]

        for kernel_width in kernel_widths:
            lime_berk_evaluate(model, data['X_train'], data['X_test'], data['y_test'], w, threshold, data_indices, actionable_indices, categorical_features, weight_dir, kernel_width)

        # outputs results in
        compute_threshold_upperbounds(model, data, w, delta_max, actionable_indices, epsilons, d, weight_dir)


if __name__ == '__main__':
    delta_max = 0.75
    data = "compas" # one of ["adult", "compas", "bail"]
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # lambda values
    kernel_widths = [0.5]
    epsilons = [0.95]
    d = 0.95
    main(data, delta_max, weights, kernel_widths, epsilons, d)
