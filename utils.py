import tensorflow as tf
from alibi.explainers import CounterFactual
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import torch
import numpy as np
import pandas as pd
from train_utils import *
from small_model import *
from big_model import *
import logging
from compute_threshold_upperbound_utils import *
from lime import lime_tabular
from tensorflow import keras

from lime import lime_tabular
from recourse.action_set import ActionSet
from recourse.flipset import Flipset

def load_torch_model(weight_dir, weight):
    model_name = weight_dir + str(weight) + '_best_model.pt'
    model = torch.load(model_name)
    model.eval()
    return model

def pred_function(model, np_x):
    torch_x = torch.from_numpy(np_x).float()
    model_pred = model(torch_x).detach().numpy()
    probs = np.concatenate((model_pred, 1-model_pred), axis=1)
    return probs

def print_test_results(file_name, model, threshold, weight, data, labels, precision):
    """
    prints test metrics (for wachter_evaluate and our_evaluate) in file file_name

    :param file_name: name of file within experiment_dir/model_dir/test_eval folder to write test metrics
    :param model: model to evaluate
    :param threshold
    :param weight
    :param data

    :returns: 
    """

    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)
    
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    
    f1 = round(f1_score(y_true, y_pred), 3)
    recall = round(recall_score(y_true, y_pred), 3)
    acc = round(np.sum(y_pred == y_true)/(y_true).shape[0], 3)

    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds
    
    f = open(file_name, "w")
    
    print("WEIGHT: {}".format(weight), file=f)
    print("THRESHOLD: {}".format(threshold), file=f)
    print("# pos preds: ", pos_preds, file=f)
    print("# neg preds: ", neg_preds, file=f)
    print("f1: ", f1, file=f)
    print("precision: ", precision, file=f)
    print("recall: ", recall, file=f)
    print("acc: ", acc, file=f)
    f.close()

    return f1, recall, acc

def predict_as_numpy(model):
    def new_predict_proba(data):
        return model(torch.from_numpy(data).float()).detach().numpy()
    return new_predict_proba   

def get_lime_coefficients(lime_exp, categorical_features, num_features):
    coefficients = [None] * num_features
    tuple_coefficients = lime_exp.as_list()
    for (feat, coef) in tuple_coefficients:
        int_feat = int(feat.split("=")[0])
        if int_feat in categorical_features:
            value = int(feat[-1])
            if value == 0:
                coef = coef * -1
            else:
                coef = coef
        coefficients[int_feat] = coef
    return coefficients


def lime_berk_evaluate(model, X_train, X_test, y_test, weight, threshold, data_indices, actionable_indices, categorical_features, model_dir, kernel_width):

    test_eval_dir = model_dir + "test_eval/lime_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_berk-lime-' + str(kernel_width) + '_' + str(threshold) + '_test.txt'

    model.eval()
    data = X_test.iloc[data_indices]    
    labels = y_test.iloc[data_indices]


    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)

    # this part is redundant with print_test_results
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    precision = round(precision_score(y_true, y_pred), 3)
    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds

    f1, recall, acc = print_test_results(file_name, model, threshold, weight, data, labels, precision)

    pred_fn = predict_as_numpy(model)

    lime_explainer = lime_tabular.LimeTabularExplainer(data.values, mode="regression", \
                                categorical_features=categorical_features, discretize_continuous=False, \
                                kernel_width = kernel_width, feature_selection='none')

    action_set = ActionSet(X_train)

    for feat_idx, feat in enumerate(data):
        if feat_idx not in actionable_indices:
            action_set[feat].mutable = False

    f = open(file_name, "a")
    print("kernel width: {}".format(kernel_width), file=f)
    print("len(instances) evaluated on: {}\n\n".format(len(data_indices)), file=f)
    f.close()

    scores = pred_fn(data.values)

    neg_test_preds = np.nonzero(scores < threshold)[0]

    num_neg_instances = 0
    num_errors = 0
    flipped = 0

    weight_dir = model_dir

    logger = logging.getLogger('recourse.builder')
    logging.basicConfig(level=logging.CRITICAL)



    for i in tqdm(neg_test_preds, total = len(neg_test_preds)):

        sample = data.iloc[i].values
    
        num_neg_instances += 1
        exp = lime_explainer.explain_instance(sample, pred_fn)
        coefficients = get_lime_coefficients(exp, categorical_features, len(sample))
        intercept = exp.intercept[0]

        # subtract bc flipset treats > 0 as positive and < 0 as negative
        intercept = intercept - threshold

        action_set.align(coefficients=coefficients)

        fb = Flipset(x = sample, action_set = action_set, coefficients = coefficients, intercept = intercept)

        try:
            fb = fb.populate(enumeration_type = 'distinct_subsets', total_items = 1)
            action = (fb.items[0]['actions'])
            if pred_fn(sample + action)[0] > threshold:
                flipped += 1
        except:
            print("exception")
            num_errors += 1
    
    num_with_recourses = num_neg_instances - num_errors

    num_pos_instances = len(data.values) - num_neg_instances


    if num_neg_instances != 0:
        flipped_proportion = round((flipped)/num_neg_instances, 3)
        none_returned_proportion = round(num_errors/num_neg_instances, 3)
    else:
        flipped_proportion = 0
        none_returned_proportion = 0

    recourse_fraction = round((num_pos_instances + flipped)/len(scores), 3)
    
    assert((num_pos_instances + num_neg_instances) == len(scores))
    
    f = open(file_name, "a")
    print("num none returned (errors): {}/{}, {}".format(num_errors, num_neg_instances, none_returned_proportion), file=f)
    print("flipped: {}/{}, {}".format((flipped), num_neg_instances, flipped_proportion), file=f)
    print("proportion with recourse: {}".format(recourse_fraction), file=f)
    print("--------\n\n", file=f) 
    f.close()

    return flipped_proportion, precision, recourse_fraction, f1, recall, acc


def compute_threshold_upperbounds(model, X_test, y_test, weight, delta_max, data_indices, actionable_indices, epsilons, d, model_dir):
    """
    calculate the optimal delta using linear program

    :param model: pytorch model to evaluate
    :param X_test: X test data (e.g. adult_data['X_test'])
    :param y_test: y test data (e.g. adult_data['y_test'])
    :param weight: weight being evaluated (used to name file)
    :param threshold: threshold to use in evaluation
    :param delta_max: parameter defining maximum change in individual feature value
    :actionable_indices: indices of actionable features
    :model_dir: model (weight) specific directory within experiment directory

    :returns: 
    """

    test_eval_dir = model_dir + "test_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_our_threshold_bounds.csv'

    model.eval()

    data = X_test.iloc[data_indices]    
    labels = y_test.iloc[data_indices]

    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)

    probs = []

    for i in range(len(torch_labels)):
        x = torch_data[i]              # data point
        
        delta_opt = calc_delta_opt(model, x, delta_max, actionable_indices)
        probs.append(model(x + delta_opt).item())

    probs = np.array(probs)

    threshold_bounds = []
    ds = []

    for epsilon in epsilons:
        t = compute_t(probs, 1 - epsilon, d)
        threshold_bounds.append(t)
        ds.append(d)

        print(t)

    thresholds_data = {}
    thresholds_data['threshold_bounds'] = threshold_bounds
    thresholds_data['epsilons'] = epsilons
    thresholds_data['d'] = ds

    thresholds_df = pd.DataFrame(data=thresholds_data)
    thresholds_df['threshold_bounds'] = thresholds_df['threshold_bounds'].round(3)

    print(threshold_bounds)

    thresholds_df.to_csv(file_name, index_label='index')


def our_evaluate(model, X_test, y_test, weight, threshold, delta_max, data_indices, actionable_indices, model_dir, do_print_individual_files = True):
    """
    calculate the optimal delta using linear program

    :param model: pytorch model to evaluate
    :param X_test: X test data (e.g. adult_data['X_test'])
    :param y_test: y test data (e.g. adult_data['y_test'])
    :param weight: weight being evaluated (used to name file)
    :param threshold: threshold to use in evaluation
    :param delta_max: parameter defining maximum change in individual feature value
    :actionable_indices: indices of actionable features
    :model_dir: model (weight) specific directory within experiment directory

    :returns: 
    """

    test_eval_dir = model_dir + "test_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_our_' + str(threshold) + '_test.txt'

    model.eval()

    #TODO: this is hacky; but we only need this part for minority evaluate
    if data_indices != None:
        data = X_test.iloc[data_indices]    
        labels = y_test.iloc[data_indices]

    else:
        data = X_test
        labels = y_test

    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)

    # this part is redundant with print_test_results
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    precision = round(precision_score(y_true, y_pred), 3)
    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds

    f1, recall, acc = print_test_results(file_name, model, threshold, weight, data, labels, precision)

    if do_print_individual_files:
        f = open(file_name, "a")
        print("DELTA MAX: {}".format(delta_max), file=f)
        print("len(instances) evaluated on: {}\n\n".format(len(data_indices)), file=f)

    loss_fn = torch.nn.BCELoss()

    negative_instances, flipped = 0, 0
    total_instances = len(labels)

    for i in range(len(torch_labels)):
        x = torch_data[i]              # data point
        y_pred = model(x).item()
        
        if y_pred < threshold: #negative pred
            negative_instances += 1
            delta_opt = calc_delta_opt(model, x, delta_max, actionable_indices)
            if model(x + delta_opt) > threshold:
                flipped += 1

    recourse_fraction = round((flipped + pos_preds)/total_instances, 3)

    if negative_instances != 0:
        flipped_proportion = round((flipped)/negative_instances, 3)
    else:
        flipped_proportion = 0

    if do_print_individual_files:
        f.write("\nlen(test): {}".format(len(labels)))
        if negative_instances != 0:
            f.write("\nflipped/negative: {}, {}/{}".format(flipped_proportion, flipped, negative_instances))
        else:
            f.write("\nflipped/negative: {}, {}/{}".format('NA', flipped, negative_instances))
        f.write("\nportion of all instances with recourse: {} (includes pos preds)".format(recourse_fraction))
        f.write('\n\nBASELINE STATS: ')
        maj_label = 0.0
        min_label = 1.0
        maj_baseline_preds = (np.full(y_true.shape, maj_label)).ravel().tolist()
        min_baseline_preds = (np.full(y_true.shape, min_label)).ravel().tolist()
        f.write("\ntest maj ({}) baseline accuracy: {}\n".format(maj_label, round(np.sum(maj_baseline_preds == y_true)/(y_true).shape[0], 3)))
        f.write("test min ({}) baseline accuracy: {}\n".format(min_label, round(np.sum(min_baseline_preds == y_true)/(y_true).shape[0], 3)))
        f.write("test min baseline f1: {}\n\n".format(round(f1_score((y_true).ravel().tolist(), min_baseline_preds), 3)))            
        f.close()    

    return flipped_proportion, precision, recourse_fraction, f1, recall, acc
        
def get_threshold_info(model_dir, weight):
    # this is based on the value of this name in the train function
    best_model_thresholds_file_name = model_dir + str(weight) + '_val_thresholds_info.csv'
    threshold_df = pd.read_csv(best_model_thresholds_file_name, dtype=np.float64, index_col = 'index')
    return threshold_df

def write_threshold_info(model_dir, weight, thresholds_file_name, thresholds, f1s, accuracies, precisions, recalls, flipped_proportion, recourse_proportion):
    thresholds_data = {}
    thresholds_data['thresholds'] = thresholds
    thresholds_data['f1s'] = f1s
    thresholds_data['accuracies'] = accuracies
    thresholds_data['precisions'] = precisions
    thresholds_data['recalls'] = recalls
    thresholds_data['flipped_proportion'] = flipped_proportion
    thresholds_data['recourse_proportion'] = recourse_proportion

    thresholds_df = pd.DataFrame(data=thresholds_data)
    thresholds_df['thresholds'] = thresholds_df['thresholds'].round(3)

    thresholds_df = thresholds_df[['thresholds', 'f1s', 'accuracies', 'precisions', 'recalls', 'flipped_proportion', 'recourse_proportion']]

    thresholds_df.to_csv(thresholds_file_name, index_label='index')

def run_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, \
    thresholds = None, lam_init = 0.005, max_lam_steps = 50, data_indices = range(0, 500)):
    """
    Runs wachter + our evaluation for every threshold in the 'WEIGHT_val_thresholds_info.csv' file output by the train function

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
    wachter_thresholds_file_name = model_dir + "test_eval/" + "wachter_thresholds_test_results.csv"

    # name of file where to output all results for different thresholds
    our_thresholds_file_name = model_dir + "test_eval/" + "our_thresholds_test_results.csv"

    # lists in which to store results for diff thresholds
    wachter_thresholds, wachter_precisions, wachter_flipped_proportions, wachter_recourse_proportions, wachter_f1s, wachter_recalls, wachter_accs = [], [], [], [], [], [], []
    our_thresholds, our_precisions, our_flipped_proportions, our_recourse_proportions, our_f1s, our_recalls, our_accs = [], [], [], [], [], [], []

    # thresholds arg not supplied, read in thresholds from validation evaluation during training and use those
    if thresholds == None:
        threshold_df = get_threshold_info(model_dir, w)
        thresholds = list(threshold_df['thresholds'])

    print("THRESHOLDS: ", thresholds)
    for threshold in thresholds:
        threshold = round(threshold, 3)
        print("THR: ", threshold)
        our_flipped_proportion, our_precision, our_recourse_fraction, our_f1, our_recall, our_acc = our_evaluate(model, data['X_test'], data['y_test'], w, threshold, delta_max, data_indices, actionable_indices, model_dir)
        our_thresholds.append(threshold)
        our_precisions.append(our_precision)
        our_flipped_proportions.append(our_flipped_proportion)
        our_recourse_proportions.append(our_recourse_fraction)
        our_f1s.append(our_f1)
        our_recalls.append(our_recall)
        our_accs.append(our_acc)

        wachter_flipped_proportion, wachter_precision, wachter_recourse_fraction, wachter_f1, wachter_recall, wachter_acc = wachter_evaluate(model, data['X_test'], data['y_test'], w, threshold, delta_max, lam_init, max_lam_steps, data_indices, actionable_indices, model_dir)
        wachter_thresholds.append(threshold)
        wachter_precisions.append(wachter_precision)
        wachter_flipped_proportions.append(wachter_flipped_proportion)
        wachter_recourse_proportions.append(wachter_recourse_fraction)
        wachter_f1s.append(wachter_f1)
        wachter_recalls.append(wachter_recall)
        wachter_accs.append(wachter_acc)

    write_threshold_info(model_dir, w, wachter_thresholds_file_name, wachter_thresholds, wachter_f1s, wachter_accs, wachter_precisions, wachter_recalls, wachter_flipped_proportions, wachter_recourse_proportions)
    write_threshold_info(model_dir, w, our_thresholds_file_name, our_thresholds, our_f1s, our_accs, our_precisions, our_recalls, our_flipped_proportions, our_recourse_proportions)
                    
def wachter_evaluate(model, X_test, y_test, weight, threshold, delta_max, lam_init, max_lam_steps, data_indices, actionable_indices, model_dir, do_print_individual_files = True):
    """
    calculate the optimal delta using linear program

    :param model: pytorch model to evaluate
    :param X_test: X test dataframe data (e.g. adult_data['X_test'])
    :param y_test: y test dataframe data (e.g. adult_data['y_test'])
    :param weight: weight being evaluated (used to name file)
    :param threshold: threshold to use in evaluation
    :param delta_max: parameter defining maximum change in individual feature value
    :param lam_init: lam hyperparameter for wachter evaluation
    :data_indices: subset of data to evaluate on (e.g. something like range(100, 300))
    :actionable_indices: indices of actionable features
    :model_dir: model (weight) specific directory within experiment directory

    :returns: 
    """

    test_eval_dir = model_dir + "test_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_wachter_' + str(threshold) + '_test.txt'

    model.eval()
    data = X_test.iloc[data_indices]    
    labels = y_test.iloc[data_indices]

    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)

    # this part is redundant with print_test_results
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    precision = round(precision_score(y_true, y_pred), 3)
    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds

    f1, recall, acc = print_test_results(file_name, model, threshold, weight, data, labels, precision)

    if do_print_individual_files:
        f = open(file_name, "a")
        print("LAM INIT: {}".format(lam_init), file=f)
        print("MAX LAM STEPS: {}".format(max_lam_steps), file=f)
        print("DELTA MAX: {}".format(delta_max), file=f)
        print("len(instances) evaluated on: {}\n\n".format(len(data_indices)), file=f)
        f.close()

    # consider all lams from lam_init/10 to lam_init; 
    #if cannot find any counterfactual instances using those values, i am assuming it is an unsolvable problem

    do_print = True 

    scores = (pred_function(model, data.values)[:, 0])

    neg_test_preds = np.nonzero(np.array(scores) < threshold)[0]

    num_no_recourses = 0
    num_neg_instances = 0
    num_none_returned = 0


    weight_dir = model_dir

    logger = logging.getLogger('alibi.explainers.counterfactual')
    logging.basicConfig(level=logging.CRITICAL)

    for i in tqdm(neg_test_preds, total = len(neg_test_preds)):

        sample = data.iloc[i].values.reshape(1,-1)
        mins = sample[0].copy()
        maxs = sample[0].copy()
        for ai in actionable_indices:
            mins[ai] = mins[ai] - delta_max
            maxs[ai] = maxs[ai] + delta_max
        mins = mins.reshape(1,-1)
        maxs = maxs.reshape(1,-1)


        tol = (1 - threshold)
        tf.compat.v1.disable_eager_execution()
        tf.keras.backend.clear_session()
        explainer = CounterFactual(lambda x: pred_function(model, x), \
                               shape=(1,) + data.iloc[0].values.shape, target_proba = 1.0, \
                               target_class='other', tol = tol, feature_range = (mins, maxs), \
                               lam_init = lam_init, max_lam_steps = max_lam_steps)
        try:
            recourse = explainer.explain(sample)
            if recourse.cf != None:
                action = (recourse.cf['X'][0]) - sample
                if do_print:
                    print("lambda: ", recourse.cf['lambda'])
                    print('index: ', recourse.cf['index'])
                    print("action: ", np.around(action, 2))
                    print("sample: ", sample)
                    print("counterfactual: ", recourse.cf['X'][0])
                    print("counterfactual proba: ", recourse.cf['proba'])
                    print("normal proba: ", pred_function(model, sample))
            else:
                num_no_recourses += 1
                num_none_returned += 1
        except UnboundLocalError as e:
            num_no_recourses += 1
            if do_print:
                print(e)
                print("no success")
        num_neg_instances += 1
        
    num_with_recourses = num_neg_instances - num_no_recourses

    if num_neg_instances != 0:
        flipped_proportion = round((num_neg_instances-num_no_recourses)/num_neg_instances, 3)
        none_returned_proportion = round(num_none_returned/num_neg_instances, 3)
    else:
        flipped_proportion = 0
        none_returned_proportion = 0

    recourse_fraction = round((pos_preds + num_with_recourses)/len(y_pred), 3)
    
    assert(num_neg_instances == neg_preds)
    assert((pos_preds + num_neg_instances) == len(y_pred))
    
    if do_print_individual_files:
        f = open(file_name, "a")
        print("num none returned: {}/{}, {}".format(num_none_returned, num_neg_instances, none_returned_proportion), file=f)
        print("flipped: {}/{}, {}".format((num_neg_instances - num_no_recourses), num_neg_instances, flipped_proportion), file=f)
        print("proportion with recourse: {}".format(recourse_fraction), file=f)
        print("--------\n\n", file=f) 
        f.close()

    return flipped_proportion, precision, recourse_fraction, f1, recall, acc

def tf_wachter_evaluate(model, X_test, y_test, weight, threshold, delta_max, lam_init, max_lam_steps, data_indices, actionable_indices, model_dir):
    """
    calculate the optimal delta using linear program

    :param model: pytorch model to evaluate
    :param X_test: X test dataframe data (e.g. adult_data['X_test'])
    :param y_test: y test dataframe data (e.g. adult_data['y_test'])
    :param weight: weight being evaluated (used to name file)
    :param threshold: threshold to use in evaluation
    :param delta_max: parameter defining maximum change in individual feature value
    :param lam_init: lam hyperparameter for wachter evaluation
    :data_indices: subset of data to evaluate on (e.g. something like range(100, 300))
    :actionable_indices: indices of actionable features
    :model_dir: model (weight) specific directory within experiment directory

    :returns: 
    """

    test_eval_dir = model_dir + "test_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_wachter_' + str(threshold) + '_test.txt'

    model.eval()
    data = X_test.iloc[data_indices]    
    labels = y_test.iloc[data_indices]

    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)

    # this part is redundant with print_test_results
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    precision = round(precision_score(y_true, y_pred), 3)
    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds

    f1, recall, acc = print_test_results(file_name, model, threshold, weight, data, labels, precision)

    f = open(file_name, "a")
    print("LAM INIT: {}".format(lam_init), file=f)
    print("DELTA MAX: {}".format(delta_max), file=f)
    print("len(instances) evaluated on: {}\n\n".format(len(data_indices)), file=f)
    f.close()

    # consider all lams from lam_init/10 to lam_init; 
    #if cannot find any counterfactual instances using those values, i am assuming it is an unsolvable problem

    do_print = True 

    scores = (pred_function(model, data.values)[:, 0])

    neg_test_preds = np.nonzero(np.array(scores) < threshold)[0]

    num_no_recourses = 0
    num_neg_instances = 0
    num_none_returned = 0


    weight_dir = model_dir

    logger = logging.getLogger('alibi.explainers.counterfactual')
    logging.basicConfig(level=logging.CRITICAL)

    dummy_input = torch.from_numpy(data.values[0:2]).float()
    onxx_model_name = save_model_as_onxx(model, dummy_input, weight_dir, weight)
    onnx_model = onnx.load(onxx_model_name)

    tf.compat.v1.disable_eager_execution()

    k_model = onnx_to_keras(onnx_model, ['input.1'])

    keras_model_name = weight_dir + str(weight) + '_best_model.h5'

    k_model.save(keras_model_name)

    tf.keras.backend.clear_session()    

    new_k_model = keras.models.load_model(keras_model_name)

    tf.compat.v1.disable_eager_execution()
    new_k_model._make_predict_function()

    global graph
    graph = tf.compat.v1.get_default_graph()

    for i in tqdm(neg_test_preds, total = len(neg_test_preds)):
        tf.compat.v1.reset_default_graph()
        with graph.as_default():

            sample = data.iloc[i].values.reshape(1,-1)
            mins = sample[0].copy()
            maxs = sample[0].copy()
            for ai in actionable_indices:
                mins[ai] = mins[ai] - delta_max
                maxs[ai] = maxs[ai] + delta_max
            mins = mins.reshape(1,-1)
            maxs = maxs.reshape(1,-1)


            original_pred = new_k_model.predict(sample).item()
            original_pred = 1 if original_pred > threshold else 0
            target_pred = 1 if original_pred == 0 else 0
            tol = (1 - threshold)

            explainer = CounterFactual(new_k_model, \
                                   shape=(1,) + data.iloc[0].values.shape, target_proba = 1.0, \
                                   tol=tol, target_class='same', feature_range = (mins, maxs), \
                                   lam_init = lam_init, max_lam_steps = max_lam_steps)
            try:
                recourse = explainer.explain(sample)
                if recourse.cf != None:
                    action = (recourse.cf['X'][0]) - sample
                    if do_print:
                        print("lambda: ", recourse.cf['lambda'])
                        print('index: ', recourse.cf['index'])
                        print("action: ", np.around(action, 2))
                        print("sample: ", sample)
                        print("counterfactual: ", recourse.cf['X'][0])
                        print("counterfactual proba: ", recourse.cf['proba'])
                        print("normal proba: ", new_k_model.predict(sample))
                else:
                    num_no_recourses += 1
                    num_none_returned += 1
            except UnboundLocalError as e:
                num_no_recourses += 1
                if do_print:
                    print(e)
                    print("no success")
        num_neg_instances += 1
        
    num_with_recourses = num_neg_instances - num_no_recourses

    if num_neg_instances != 0:
        flipped_proportion = round((num_neg_instances-num_no_recourses)/num_neg_instances, 3)
        none_returned_proportion = round(num_none_returned/num_neg_instances, 3)
    else:
        flipped_proportion = 0
        none_returned_proportion = 0

    recourse_fraction = round((pos_preds + num_with_recourses)/len(y_pred), 3)
    
    assert(num_neg_instances == neg_preds)
    assert((pos_preds + num_neg_instances) == len(y_pred))
    
    f = open(file_name, "a")
    print("num none returned: {}/{}, {}".format(num_none_returned, num_neg_instances, none_returned_proportion), file=f)
    print("flipped: {}/{}, {}".format((num_neg_instances - num_no_recourses), num_neg_instances, flipped_proportion), file=f)
    print("proportion with recourse: {}".format(recourse_fraction), file=f)
    print("--------\n\n", file=f) 
    f.close()

    return flipped_proportion, precision, recourse_fraction, f1, recall, acc

def run(data, actionable_indices, experiment_dir, weights, do_train, lam_init = 0.001, max_lam_steps = 10):
    
    lr = 0.002 # changed this for compas training
    delta_max = 0.75
    fixed_precisions = [0.4, 0.5, 0.6, 0.7]
    fixed_precisions = []

    weight_dir = experiment_dir + str(w) + "/"

    for w in weights:
        print("WEIGHT: ", w)
        print(data['X_train'].values[0])
        model = SmallModel(len(data['X_train'].values[0]))
        
        torch_X_train = torch.from_numpy(data['X_train'].values).float()
        torch_y_train = torch.from_numpy(data['y_train'].values).float()
        torch_X_val = torch.from_numpy(data['X_val'].values).float()
        torch_y_val = torch.from_numpy(data['y_val'].values).float()
        
        if do_train:
            # train the model
            train(model, torch_X_train, torch_y_train, \
                 torch_X_val, torch_y_val, actionable_indices, experiment_dir, \
                  recourse_loss_weight = w, num_epochs = 1, delta_max = delta_max, lr=lr, \
                  fixed_precisions = fixed_precisions)
        
        else:
            model = load_torch_model(weight_dir, weight)

        run_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, lam_init = lam_init, max_lam_steps = max_lam_steps)


def run_minority_evaluate(model, dict_data, w, delta_max, actionable_indices, experiment_dir, white_feature_name, \
    thresholds = None, lam_init = 0.005, max_lam_steps = 50, data_indices = range(0, 500)):

    # define the data indices to consider
    model_dir = experiment_dir + str(w) + "/"

    data = dict_data['X_test'].iloc[data_indices]  
    labels = dict_data['y_test'].iloc[data_indices]  

    white_data = data.loc[data[white_feature_name] == 1]
    minority_data = data.loc[data[white_feature_name] == 0]

    white_labels = labels[labels.index.isin(white_data.index)]
    minority_labels = labels[labels.index.isin(minority_data.index)]

    # lists in which to store results for diff thresholds
    wachter_thresholds, wachter_precisions, wachter_flipped_proportions, wachter_recourse_proportions, wachter_f1s, wachter_recalls, wachter_accs = [], [], [], [], [], [], []
    our_thresholds, our_precisions, our_flipped_proportions, our_recourse_proportions, our_f1s, our_recalls, our_accs = [], [], [], [], [], [], []

    print("THRESHOLDS: ", thresholds)
    # WHITE DATA:

    # name of file where to output all results for different thresholds
    wachter_thresholds_file_name = model_dir + "test_eval/minority_exp/" + "WHITE_wachter_thresholds_test_results.csv"

    # name of file where to output all results for different thresholds
    our_thresholds_file_name = model_dir + "test_eval/minority_exp/" + "WHITE_our_thresholds_test_results.csv"

    for threshold in thresholds:
        threshold = round(threshold, 3)
        print("THR: ", threshold)

        our_flipped_proportion, our_precision, our_recourse_fraction, our_f1, our_recall, our_acc = our_evaluate(model, white_data, white_labels, w, threshold, delta_max, None, actionable_indices, model_dir, do_print_individual_files = False)
        our_thresholds.append(threshold)
        our_precisions.append(our_precision)
        our_flipped_proportions.append(our_flipped_proportion)
        our_recourse_proportions.append(our_recourse_fraction)
        our_f1s.append(our_f1)
        our_recalls.append(our_recall)
        our_accs.append(our_acc)

        # wachter_flipped_proportion, wachter_precision, wachter_recourse_fraction, wachter_f1, wachter_recall, wachter_acc = wachter_evaluate(model, white_data, white_labels, w, threshold, delta_max, lam_init, max_lam_steps, None, actionable_indices, model_dir, do_print_individual_files = False)
        # wachter_thresholds.append(threshold)
        # wachter_precisions.append(wachter_precision)
        # wachter_flipped_proportions.append(wachter_flipped_proportion)
        # wachter_recourse_proportions.append(wachter_recourse_fraction)
        # wachter_f1s.append(wachter_f1)
        # wachter_recalls.append(wachter_recall)
        # wachter_accs.append(wachter_acc)

    # write_threshold_info(model_dir, w, wachter_thresholds_file_name, wachter_thresholds, wachter_f1s, wachter_accs, wachter_precisions, wachter_recalls, wachter_flipped_proportions, wachter_recourse_proportions)
    write_threshold_info(model_dir, w, our_thresholds_file_name, our_thresholds, our_f1s, our_accs, our_precisions, our_recalls, our_flipped_proportions, our_recourse_proportions)
                    
    # name of file where to output all results for different thresholds
    wachter_thresholds_file_name = model_dir + "test_eval/minority_exp/" + "MINORITY_wachter_thresholds_test_results.csv"

    # name of file where to output all results for different thresholds
    our_thresholds_file_name = model_dir + "test_eval/minority_exp/" + "MINORITY_our_thresholds_test_results.csv"

    for threshold in thresholds:
        threshold = round(threshold, 3)
        print("THR: ", threshold)

        our_flipped_proportion, our_precision, our_recourse_fraction, our_f1, our_recall, our_acc = our_evaluate(model, minority_data, minority_labels, w, threshold, delta_max, None, actionable_indices, model_dir, do_print_individual_files = False)
        our_thresholds.append(threshold)
        our_precisions.append(our_precision)
        our_flipped_proportions.append(our_flipped_proportion)
        our_recourse_proportions.append(our_recourse_fraction)
        our_f1s.append(our_f1)
        our_recalls.append(our_recall)
        our_accs.append(our_acc)

        # wachter_flipped_proportion, wachter_precision, wachter_recourse_fraction, wachter_f1, wachter_recall, wachter_acc = wachter_evaluate(model, white_data, white_labels, w, threshold, delta_max, lam_init, max_lam_steps, None, actionable_indices, model_dir, do_print_individual_files = False)
        # wachter_thresholds.append(threshold)
        # wachter_precisions.append(wachter_precision)
        # wachter_flipped_proportions.append(wachter_flipped_proportion)
        # wachter_recourse_proportions.append(wachter_recourse_fraction)
        # wachter_f1s.append(wachter_f1)
        # wachter_recalls.append(wachter_recall)
        # wachter_accs.append(wachter_acc)

    # write_threshold_info(model_dir, w, wachter_thresholds_file_name, wachter_thresholds, wachter_f1s, wachter_accs, wachter_precisions, wachter_recalls, wachter_flipped_proportions, wachter_recourse_proportions)
    write_threshold_info(model_dir, w, our_thresholds_file_name, our_thresholds, our_f1s, our_accs, our_precisions, our_recalls, our_flipped_proportions, our_recourse_proportions)
                    
