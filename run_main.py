from utils.data_utils import *
from utils.train_utils import *
from utils.other_utils import *
import argparse

def main(dataset, delta_max, weights, with_noise = False):
    """
    runs the main experiments in the paper

    :param data: string in ["adult", "compas", "bail"]
    :param delta_max: parameter defining maximum change in individual feature value
    :weights: lambda values determining the weight on our adversarial training objective

    """
    assert dataset in ["adult", "compas", "bail"]

    if dataset in ["adult", "compas"]:
        X, y, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, categorical_features, _, feature_names, _, _ = process_data(dataset)
        # create a test set from data
        X_test, y_test = None, None
    else:
        X_test, y_test, _, _, _, _, _, _, _ = process_bail_data(subset = "test")
        X, y, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, categorical_features, _, feature_names, _, _ = process_bail_data(subset = "train")

    if dataset == "adult":
        white_feature_name = "isWhite"
    elif dataset == "compas":
        white_feature_name = "isCaucasian"
    elif dataset == "bail":
        white_feature_name = "WHITE"

    if with_noise:
        experiment_dir = 'results/' + dataset + '_' + str(delta_max) + '_noise/'
    else:
        experiment_dir = 'results/' + dataset + '_' + str(delta_max) + '/'

    print("reading data")
    data = read_data(experiment_dir) # read data if we've already written data
    if data == None:
        data = get_data(X, y, X_test = X_test, y_test = y_test)
        print("writing data")
        write_data(data, experiment_dir)
        data = read_data(experiment_dir) # read data if we've already written data

    # ------- FURTHER EXPERIMENTS (minority disparities, LIME + linear evaluation, theoretical guarantees) ------- 
    data_indices = range(0, 500)

    for w in weights:

        print("WEIGHT: ", w)

        weight_dir = experiment_dir + str(w) + "/"

        # ------- MAIN EXPERIMENT -------
        # (training, evaluating recourse/performance metrics using gradient descent and adversarial training algorithms for computing recourse)
        run(data, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, categorical_features, experiment_dir, [w], delta_max, with_noise = with_noise, do_train = True)


        model = load_torch_model(weight_dir, w)    

        if w in [0.0, 1.0]:
            # EVAL WITH NO CONSTRAINTS (training, evaluating recourse/performance metrics using gradient descent and adversarial training algorithms for computing recourse)
            run(data, actionable_indices, [], [], categorical_features, experiment_dir, [w], delta_max, with_noise = with_noise, do_train = False, no_constraints = True)

                    # LIME LINEAR APPROXIMATION (only evaluate at the threshold that maximizes f1 score on val data)
            threshold_df = get_threshold_info(weight_dir, w)
            thresholds = list(threshold_df['thresholds'])
            f1s = threshold_df['f1s'] 
            eval_thresholds = [thresholds[np.argmax(f1s)]]
            threshold = eval_thresholds[0]

            lime_linear_evaluate(model, data['X_train'], data['X_test'], data['y_test'], w, threshold, data_indices, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, categorical_features, weight_dir)

            # THEORETICAL PARE GUARANTEES (computes metrics at thresholds satisfying theoretical upperbound derived with PARE guarantees)

            epsilons = [0.95] # parameter for theory experiment
            alpha = 0.95 # parameter for theory experiment
            compute_threshold_upperbounds(model, data, w, delta_max, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, epsilons, alpha, weight_dir)


        # MINORITY DISPARITIES (runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function)
        run_minority_evaluate(model, data, w, delta_max, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, experiment_dir, white_feature_name, lam_init = 0.001, data_indices = data_indices)


# EXAMPLE RUN
if __name__ == '__main__':
    delta_max = 0.75
    # weights = [0.7, 0.1, 0.9, 0.3,0.5, 1.1, 1.3, 1.5, 1.7, 1.9] # lambda values
    weights = [0.0, 1.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.0] # lambda values
    # weights = [0.0]

    # with_noise = False

    parser = argparse.ArgumentParser()
    parser.add_argument('-with_noise')
    parser.add_argument('-dataset')

    args = parser.parse_args()

    with_noise = True if args.with_noise == "True" else False
    dataset = args.dataset

    assert with_noise != None
    assert dataset != None

    print("dataset: ", dataset)
    print("with noise: ", with_noise)

    main(dataset, delta_max, weights, with_noise = with_noise)

