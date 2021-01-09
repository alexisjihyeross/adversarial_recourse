from utils.data_utils import *
from utils.train_utils import *
from utils.other_utils import *

def main(data, delta_max, weights):
    """
    runs the main experiments in the paper

    :param data: string in ["adult", "compas", "bail"]
    :param delta_max: parameter defining maximum change in individual feature value
    :weights: lambda values determining the weight on our adversarial training objective

    """
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
    # data = read_data(experiment_dir) # read data if we've already written data

    # ------- MAIN EXPERIMENT -------
    # (training, evaluating recourse/performance metrics using gradient descent and adversarial training algorithms for computing recourse)
    run(data, actionable_indices, increasing_actionable_indices, categorical_features, experiment_dir, weights, delta_max, do_train = True)



    # ------- FURTHER EXPERIMENTS (minority disparities, LIME + linear evaluation, theoretical guarantees) ------- 
    data_indices = range(0, 500)

    for w in weights:
        print("minority WEIGHT: ", w)    
        weight_dir = experiment_dir + str(w) + "/"
        model = load_torch_model(weight_dir, w)    

        # MINORITY DISPARITIES (runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function)
        run_minority_evaluate(model, data, w, delta_max, actionable_indices, increasing_actionable_indices, experiment_dir, white_feature_name, lam_init = 0.001, data_indices = data_indices)

        # LIME LINEAR APPROXIMATION (only evaluate at the threshold that maximizes f1 score on val data)
        threshold_df = get_threshold_info(weight_dir, w)
        thresholds = list(threshold_df['thresholds'])
        f1s = threshold_df['f1s'] 
        eval_thresholds = [thresholds[np.argmax(f1s)]]
        threshold = eval_thresholds[0]

        lime_linear_evaluate(model, data['X_train'], data['X_test'], data['y_test'], w, threshold, data_indices, actionable_indices, increasing_actionable_indices, categorical_features, weight_dir)

        # THEORETICAL PARE GUARANTEES (computes metrics at thresholds satisfying theoretical upperbound derived with PARE guarantees)

        epsilons = [0.95] # parameter for theory experiment
        alpha = 0.95 # parameter for theory experiment
        compute_threshold_upperbounds(model, data, w, delta_max, actionable_indices, increasing_actionable_indices, epsilons, alpha, weight_dir)


# EXAMPLE RUN
if __name__ == '__main__':
    delta_max = 0.75
    data = "adult" # one of ["adult", "compas", "bail"]
    weights = [0.0, 0.6, 0.7, 0.1, 0.8, 0.2, 0.9, 0.3, 1.0, 0.4, 0.5] # lambda values

    main(data, delta_max, weights)

