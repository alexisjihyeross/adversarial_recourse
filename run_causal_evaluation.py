from utils.data_utils import *
from utils.train_utils import *
from utils.other_utils import *
from recourse_methods import *
import argparse

def run_causal_evaluation(w, delta_max=0.75, random_state=0, results_folder="results/"):
    """
    runs the main experiments in the paper

    :param data: string in ["adult", "compas", "bail"]
    :param delta_max: parameter defining maximum change in individual feature value
    :weights: lambda values determining the weight on our adversarial training objective

    """
    dataset = "german"
    random_seed_subfolder = str(random_state) + '/'
    experiment_dir = results_folder + dataset + '_' + str(delta_max) + '/' + random_seed_subfolder
    weight_dir = experiment_dir + str(w) + "/"

    X, y, actionable_indices, increasing_actionable_indices, decreasing_actionable_indices, categorical_features, _, feature_names, _, _ = process_data(dataset)
    # create a test set from data
    X_test, y_test = None, None

    test_size = 100

    print("reading data")
    data = read_data(experiment_dir) # read data if we've already written data
    if data == None:
        data = get_data(X, y, X_test=X_test, y_test=y_test, random_state=random_state, test_size=test_size)
        print("writing data")
        write_data(data, experiment_dir)
        data = read_data(experiment_dir) # read data if we've already written data

    # ------- FURTHER EXPERIMENTS (minority disparities, LIME + linear evaluation, theoretical guarantees) ------- 
    data_indices = range(0, test_size)

    model = load_torch_model(weight_dir, w)    

    predict_proba = predict_as_numpy(model)

    feature_costs = None
    X_train = data['X_train']
    causal_recourse = CausalRecourse(X_train, predict_proba, model,
                feature_costs=feature_costs)

    # only evaluate at the threshold that maximizes f1 score on val data
    threshold_df = get_threshold_info(weight_dir, w)
    thresholds = list(threshold_df['thresholds'])
    f1s = list(threshold_df['f1s'])
    max_f1_idx = np.argmax(f1s)
    threshold = thresholds[max_f1_idx]

    print("THRESHOLD: ", threshold)

    print("Choosing hyperparameters using train")
    torch_X_train = numpy_to_torch(X_train)
    recourse_needed_idx_X1_train = recourse_needed(model, torch_X_train, threshold)
    recourse_needed_X1_train = X_train.iloc[recourse_needed_idx_X1_train].values


    torch_X_test = numpy_to_torch(data['X_test'])
    recourse_needed_idx_X1_test = recourse_needed(model, torch_X_test, threshold)
    recourse_needed_X1_test = X_train.iloc[recourse_needed_idx_X1_test].values

    predict_fn = predict_label(model, threshold)

    step_size, lamb = causal_recourse.choose_params(recourse_needed_X1_train, predict_fn)
    results_i = {}
    results_i["step_size"] = step_size
    results_i["lambda"] = lamb
    print("Chosen step_size:%f, lambda:%f" % (step_size, lamb))

    causal_recourse.step_size = step_size
    causal_recourse.lamb = lamb
    
    recourses=[]
    for x in tqdm(recourse_needed_X1_test):
        r = causal_recourse.get_recourse(x)
        recourses.append(r)
    results_i["recourses"] = recourses
    results_i["differences"] = [r - x for r, x in zip(recourses, recourse_needed_X1_test)]

    validity = recourse_validity(predict_fn, np.array(recourses), delta_max=0.75, originals = recourse_needed_X1_test)
    results_i["validity"] = validity
    print("Recourse validity: %f" % validity)


    cost = l1_cost(recourse_needed_X1_test, recourses)

    results_i["cost"] = cost
    print("cost: ", cost)
    out_file = open(weight_dir + "test_eval/causal_evaluation.txt", "w")
    print(results_i, file=out_file)
    out_file.close()


# EXAMPLE RUN
if __name__ == '__main__':
    delta_max = 0.75
    weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    parser = argparse.ArgumentParser()
    parser.add_argument('-seed')
    parser.add_argument('-results_folder')

    args = parser.parse_args()

    random_state = int(args.seed) if args.seed else 0
    results_folder = "results/" if args.results_folder is None else args.results_folder

    print("seed: ", random_state)
    print("writing results to: ", results_folder)
    for weight in weights:

        run_causal_evaluation(weight, delta_max=delta_max, random_state=random_state, results_folder=results_folder)

