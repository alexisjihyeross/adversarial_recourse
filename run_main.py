from utils.data_utils import *
from utils.train_utils import *
from utils.other_utils import *
import argparse

def main(dataset, delta_max, weights, with_noise=False, random_state=0, results_folder="results/", all_continuous_mutable=False):
    """
    runs the main experiments in the paper

    :param data: string in ["adult", "compas", "bail"]
    :param delta_max: parameter defining maximum change in individual feature value
    :weights: lambda values determining the weight on our adversarial training objective

    """
    assert dataset in ["adult", "compas", "bail", "german"]

    if dataset in ["adult", "compas", "german"]:
        X, y, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, categorical_features, _, feature_names, _, _ = process_data(dataset, all_continuous_mutable=all_continuous_mutable)
        # create a test set from data
        X_test, y_test = None, None
    else:
        X, y, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, categorical_features, _, feature_names, train_means, train_std = process_bail_data(subset = "train")
        X_test, y_test, _, _, _, _, _, _, _, _ = process_bail_data(subset = "test", given_means = train_means, given_std = train_std)

    if dataset == "adult":
        white_feature_name = "isWhite"
    elif dataset == "compas":
        white_feature_name = "isCaucasian"
    elif dataset == "bail":
        white_feature_name = "WHITE"

    random_seed_subfolder = str(random_state) + '/'

    if with_noise:
        experiment_dir = results_folder + dataset + '_' + str(delta_max) + '_noise/' + random_seed_subfolder
    else:
        experiment_dir = results_folder + dataset + '_' + str(delta_max) + '/' + random_seed_subfolder

    if dataset == "german":
        test_size = 100
    else:
        test_size = 500

    print("reading data")
    data = read_data(experiment_dir) # read data if we've already written data
    if data == None:
        data = get_data(X, y, X_test=X_test, y_test=y_test, random_state=random_state, test_size=test_size)
        print("writing data")
        write_data(data, experiment_dir)
        data = read_data(experiment_dir) # read data if we've already written data

    # ------- FURTHER EXPERIMENTS (minority disparities, LIME + linear evaluation, theoretical guarantees) ------- 
    data_indices = range(0, test_size)

    for w in weights:

        print("WEIGHT: ", w)

        weight_dir = experiment_dir + str(w) + "/"

        if dataset == "german":
            batch_size = 30
            num_epochs = 50
        else:
            batch_size = 10
            num_epochs = 15

        # ------- MAIN EXPERIMENT -------
        # (training, evaluating recourse/performance metrics using gradient descent and adversarial training algorithms for computing recourse)
        run(data, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, categorical_features, experiment_dir, [w], delta_max, 
            with_noise = with_noise, do_train = True, no_constraints = False, eval_data_indices = data_indices,
            num_epochs=num_epochs, batch_size=batch_size)


        model = load_torch_model(weight_dir, w)    

        if w in [0.0, 0.8]:
            # EVAL WITH NO CONSTRAINTS (training, evaluating recourse/performance metrics using gradient descent and adversarial training algorithms for computing recourse)
            run(data, actionable_indices, [], [], categorical_features, experiment_dir, [w], delta_max, with_noise = with_noise, do_train = False, no_constraints = True, eval_data_indices = data_indices,)

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

        if dataset != "german":
            # MINORITY DISPARITIES (runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function)
            run_minority_evaluate(model, data, w, delta_max, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, experiment_dir, white_feature_name, lam_init = 0.001, data_indices = data_indices)


# EXAMPLE RUN
if __name__ == '__main__':
    delta_max = 0.75
    weights = [0.0, 0.8, 0.4, 1.2, 1.6, 2.0, 0.2, 0.6, 1.4, 1.8, 1.0] # lambda values

    parser = argparse.ArgumentParser()
    parser.add_argument('-with_noise')
    parser.add_argument('-dataset')
    parser.add_argument('-seed')
    parser.add_argument('-results_folder')
    parser.add_argument('-all_continuous_mutable')

    args = parser.parse_args()

    with_noise = True if args.with_noise == "True" else False
    all_continuous_mutable = True if args.all_continuous_mutable == "True" else False

    dataset = args.dataset
    if all_continuous_mutable and dataset != "adult":
        raise ValueError

    random_state = int(args.seed) if args.seed else 0
    results_folder = "results/" if args.results_folder is None else args.results_folder

    assert with_noise != None
    assert dataset != None

    print("dataset: ", dataset)
    print("with noise: ", with_noise)
    print("seed: ", random_state)
    print("all continuous mutable: ", all_continuous_mutable)
    print("writing results to: ", results_folder)

    main(dataset, delta_max, weights, 
        with_noise = with_noise, 
        random_state=random_state, 
        results_folder=results_folder,
        all_continuous_mutable=all_continuous_mutable)

