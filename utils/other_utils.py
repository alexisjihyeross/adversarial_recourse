import tensorflow as tf
from alibi.explainers import CounterFactual
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
import os
import torch
import numpy as np
import pandas as pd
from utils.train_utils import *
from model import Model
import logging
from utils.theory_utils import *

from recourse.action_set import ActionSet
from recourse.flipset import Flipset
from lime import lime_tabular

def load_torch_model(weight_dir, weight):
    """
    loads saved model in eval mode
    :param weight_dir: directory where model is saved
    :param weight: lambda value (weights the recourse/adversarial part of our training objective)


    :returns: torch tensor with optimal delta value
    """
    model_name = weight_dir + str(weight) + '_best_model.pt'
    model = torch.load(model_name)
    model.eval()
    return model

def pred_function(model, np_x):

    # helper function to turn model prediction (single probability value) into predictions over two classes
    
    torch_x = torch.from_numpy(np_x).float()
    model_pred = model(torch_x).detach().numpy()
    probs = np.concatenate((model_pred, 1-model_pred), axis=1)
    return probs

def get_additional_metrics(file_name, model, threshold, weight, data, labels, precision, do_print_individual_files = True):
    """
    gets metrics and prints (for wachter_evaluate and our_evaluate) in file file_name

    :param file_name: name of file within experiment_dir/model_dir/test_eval folder to write test metrics
    :param model: model to evaluate
    :param threshold: threshold to evaluate at
    :param weight: lambda value (weights the recourse/adversarial part of our training objective)
    :param data: features in dataframe form
    :param labels: labels in dataframe form
    :param precision: precision of the model on the data at threshold
    :param do_print_individual_files: whether or not to print the file; otherwise, just returns the f1/recall/acc metrics
    # TO DO: this is hacky, used to compute the metrics f1/recall/acc and to print

    :returns: f1 score, recall, accuracy at threshold with model on data/labels
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
    
    if do_print_individual_files:
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
    # helper function that creates a wrapper around the model's predict function
    def new_predict_proba(data):
        return model(torch.from_numpy(data).float()).detach().numpy()
    return new_predict_proba   

def get_lime_coefficients(lime_exp, categorical_features, num_features):
    # helper function to get LIME coefficients from a lime explanation object
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


def lime_linear_evaluate(model, X_train, X_test, y_test, weight, threshold, data_indices, actionable_indices, categorical_features, model_dir, kernel_width):
    """
    uses LIME linear approximations with the framework by Ustun et al., 2018 to compute recourses

    :param model: model to evaluate
    :param X_train: train data in dataframe form (used to initialize action set)
    :param X_test: test data in dataframe form to evaluate on
    :param y_test: test labels in dataframe form
    :param weight: lambda value (weights the recourse/adversarial part of our training objective)
    :param threshold: threshold to use for evaluation
    :param data_indices: the indices of the test data to evaluate on (in case we want to only use a subset)
    :param actionable_indices: indices of actionable features
    :param categorical_features: categorical features in the dataset (affects LIME sampling)
    :param model_dir: directory where the model is stored (and where to output test results)
    :param kernel_width: parameter that determines the kernel_width of the lime explainer

    :returns: flipped_proportion, precision, recourse_fraction, f1, recall, acc
        flipped proportion: recourse neg
        recourse_fraction: recourse all
    """

    test_eval_dir = model_dir + "test_eval/lime_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_berk-lime-' + str(kernel_width) + '_' + str(threshold) + '_test.txt'

    model.eval()
    data = X_test.iloc[data_indices]    
    labels = y_test.iloc[data_indices]


    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)

    # this part is redundant with get_additional_metrics
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    precision = round(precision_score(y_true, y_pred), 3)
    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds

    f1, recall, acc = get_additional_metrics(file_name, model, threshold, weight, data, labels, precision)

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
            continue
    
    num_with_recourses = num_neg_instances - num_errors

    num_pos_instances = len(data.values) - num_neg_instances


    if num_neg_instances != 0:
        flipped_proportion = round((flipped)/num_neg_instances, 3)
        none_returned_proportion = round(num_errors/num_neg_instances, 3)
    else:
        flipped_proportion = 0
        none_returned_proportion = 0

    recourse_fraction = round((num_pos_instances + flipped)/len(scores), 3)
    
    assert (num_pos_instances + num_neg_instances) == len(scores), "scores don't add up"
    
    f = open(file_name, "a")
    print("num none returned (errors): {}/{}, {}".format(num_errors, num_neg_instances, none_returned_proportion), file=f)
    print("flipped: {}/{}, {}".format((flipped), num_neg_instances, flipped_proportion), file=f)
    print("proportion with recourse: {}".format(recourse_fraction), file=f)
    print("--------\n\n", file=f) 
    f.close()

    return flipped_proportion, precision, recourse_fraction, f1, recall, acc

def compute_threshold_upperbounds(model, dict_data, weight, delta_max, actionable_indices, epsilons, alpha, model_dir):
    """
    computes threshold upperbounds for PARE guarantees using our adversarial training method of calculating recourse
    evalutes at thresholds in 10 equally spaced increments below that upperbound
    prints results to a file {}_our_threshold_bounds.csv

    :param model: pytorch model to evaluate
    :param dict_data: data in dictionary form (where each value is a dataframe)
    :param weight: weight/lambda value being evaluated (used to create output file name)
    :param delta_max: parameter defining maximum change in individual feature value
    :param actionable_indices: indices of actionable features
    :param epsilons: epsilon values to evaluate at
    :param alpha: the probability that the guarantee holds
    :model_dir: model (weight) specific directory within experiment directory
    """

    test_eval_dir = model_dir + "test_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_our_threshold_bounds.csv'

    model.eval()


    X_val, y_val = dict_data['X_val'], dict_data['y_val']
    data = X_val  
    labels = y_val

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
    test_f1s = []
    test_accs = []
    test_precisions = []
    test_recalls = []
    max_thresh = []
    test_recourse_proportions = []

    for epsilon in epsilons:
        t = compute_t(probs, 1 - epsilon, alpha)
        threshold_bounds.append(t)
        ds.append(d)

        max_f1 = 0
        max_t = None
        for t_cand in np.linspace(0, t, 10):

            y_pred = [0.0 if a < t_cand else 1.0 for a in (model(torch_data).detach().numpy())]
            y_true = ((torch_labels).detach().numpy())
            y_prob_pred = model(torch_data).detach().numpy()
        
            f1 = round(f1_score(y_true, y_pred), 3)     

            if f1 > max_f1:
                max_f1 = f1
                max_t = round(t_cand, 3)

        test_torch_data = torch.from_numpy(dict_data['X_test'].values).float()
        test_labels = torch.from_numpy(dict_data['y_test'].values).float()

        flipped = 0
        for i in range(len(test_labels)):
            x = test_torch_data[i]              # data point
            
            if model(x).item() < max_t:
                delta_opt = calc_delta_opt(model, x, delta_max, actionable_indices)
                if (model(x + delta_opt).item()) >= max_t:
                    flipped += 1

        y_pred = [0.0 if a < max_t else 1.0 for a in (model(test_torch_data).detach().numpy())]
        y_true = ((test_labels).detach().numpy())
        num_pos = sum(y_pred)
        test_f1 = round(f1_score(y_true, y_pred), 3)  
        test_acc = round(np.sum(y_pred == y_true)/(y_true).shape[0], 3)
        test_prec = round(precision_score(y_true, y_pred), 3)  
        test_recall = round(recall_score(y_true, y_pred), 3)  
        test_recourse_proportion = round((num_pos + flipped)/len(test_labels), 3)

        max_thresh.append(max_t)
        test_f1s.append(test_f1)
        test_accs.append(test_acc)
        test_precisions.append(test_prec)
        test_recalls.append(test_recall)
        test_recourse_proportions.append(test_recourse_proportion)



    thresholds_data = {}
    thresholds_data['threshold_bounds'] = threshold_bounds
    thresholds_data['epsilons'] = epsilons
    thresholds_data['d'] = ds
    thresholds_data['test_f1s'] = test_f1s
    thresholds_data['test_accs'] = test_accs
    thresholds_data['test_precisions'] = test_precisions
    thresholds_data['test_recalls'] = test_recalls
    thresholds_data['test_recourse_proportions'] = test_recourse_proportions


    thresholds_data['actual_thresh'] = max_thresh

    thresholds_df = pd.DataFrame(data=thresholds_data)
    thresholds_df['threshold_bounds'] = thresholds_df['threshold_bounds'].round(3)

    print(threshold_bounds)

    thresholds_df.to_csv(file_name, index_label='index')


def our_evaluate(model, X_test, y_test, weight, threshold, delta_max, data_indices, actionable_indices, model_dir, do_print_individual_files = True, file_name = None):
    """
    evaluates the model for recourse/performance metrics using the adversarial training algorithm for computing recourse

    :param model: pytorch model to evaluate
    :param X_test: X test data (e.g. adult_data['X_test'])
    :param y_test: y test data (e.g. adult_data['y_test'])
    :param weight: lambda value being evaluated (used to name file)
    :param threshold: threshold to use in evaluation
    :param delta_max: parameter defining maximum change in individual feature value
    :param data_indices: indices of data subset to evaluate on
    :param actionable_indices: indices of actionable features
    :model_dir: model (weight) specific directory within experiment directory

    """

    test_eval_dir = model_dir + "test_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    if file_name == None:
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

    # this part is redundant with get_additional_metrics
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    precision = round(precision_score(y_true, y_pred), 3)
    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds

    f1, recall, acc = get_additional_metrics(file_name, model, threshold, weight, data, labels, precision, do_print_individual_files = do_print_individual_files)

    if do_print_individual_files:
        f = open(file_name, "a")
        print("DELTA MAX: {}".format(delta_max), file=f)
        print("len(instances) evaluated on: {}\n\n".format(len(data_indices)), file=f)

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

    # helper function to read threshold info from a csv file output during training/validation

    best_model_thresholds_file_name = model_dir + str(weight) + '_val_thresholds_info.csv'
    threshold_df = pd.read_csv(best_model_thresholds_file_name, dtype=np.float64, index_col = 'index')
    return threshold_df

def write_threshold_info(model_dir, weight, thresholds_file_name, thresholds, f1s, accuracies, precisions, recalls, flipped_proportion, recourse_proportion):
    
    # helper function to write metrics at different thresholds to a csv file

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

def run_evaluate(model, data, w, delta_max, actionable_indices, categorical_features, \
    experiment_dir, thresholds = None, lam_init = 0.001, max_lam_steps = 10, \
    data_indices = range(0, 500), only_eval_at_max_f1 = False):
    """
    Runs evaluation for both the gradient descent + adversarial training algorithms
        at every threshold in the '{}_val_thresholds_info.csv' file output by the train function 
        OR the threshold with max f1 score on validation data if only_eval_at_max_f1

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

    # only_eval_at_max_f1 can only be true if we are evaluating at thresholds from validation evaluation
    assert not (thresholds != None and only_eval_at_max_f1)

    # thresholds arg not supplied, read in thresholds from validation evaluation during training and use those
    if thresholds == None:
        threshold_df = get_threshold_info(model_dir, w)
        thresholds = list(threshold_df['thresholds'])

    f1s = []
    if only_eval_at_max_f1:
        f1s = threshold_df['f1s'] 

        # only evaluate at the threshold that maximizes f1 score on val data
        eval_thresholds = [thresholds[np.argmax(f1s)]]

    else:
        eval_thresholds = thresholds

    print("THRESHOLDS: ", thresholds)
    for threshold in eval_thresholds:
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
    evaluates the model for recourse/performance metrics using the gradient descent algorithm for computing recourse

    :param model: pytorch model to evaluate
    :param X_test: X test data (e.g. adult_data['X_test'])
    :param y_test: y test data (e.g. adult_data['y_test'])
    :param weight: lambda value being evaluated (used to name file)
    :param threshold: threshold to use in evaluation
    :param delta_max: parameter defining maximum change in individual feature value
    :param lam_init: lam_init parameter passed to the gradient descent algorithm
    :param max_lam_steps: max_lam_steps parameter passed to the gradient descent algorithm
    :param data_indices: indices of data subset to evaluate on
    :param actionable_indices: indices of actionable features
    :model_dir: model (weight) specific directory within experiment directory

    """

    test_eval_dir = model_dir + "test_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_wachter_' + str(threshold) + '_test.txt'

    model.eval()

    if data_indices != None:
        data = X_test.iloc[data_indices]    
        labels = y_test.iloc[data_indices]

    else:
        data = X_test
        labels = y_test

    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)

    # this part is redundant with get_additional_metrics
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    precision = round(precision_score(y_true, y_pred), 3)
    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds

    f1, recall, acc = get_additional_metrics(file_name, model, threshold, weight, data, labels, precision, do_print_individual_files = do_print_individual_files)

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
                assert(action <= delta_max).all()

            else:
                num_no_recourses += 1
                num_none_returned += 1
        except UnboundLocalError as e:
            num_no_recourses += 1
            if do_print:
                print(e)
                print("no success")
        except AssertionError as e:
            print("assertion error")
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

def run(data, actionable_indices, categorical_features, experiment_dir, weights, delta_max, do_train = False, lam_init = 0.001, max_lam_steps = 10, thresholds_to_eval = None):
    """
    runs the main experiment
    trains model & calls function run_evaluate (which runs recourse/performance evaluation on test data using both the adversarial training and gradient descent algorithms for computing recourse)
    """

    lr = 0.002 # changed this for compas training

    for w in weights:
        weight_dir = experiment_dir + str(w) + "/"
        print("WEIGHT: ", w)
        print(data['X_train'].values[0])
        model = Model(len(data['X_train'].values[0]))
        
        torch_X_train = torch.from_numpy(data['X_train'].values).float()
        torch_y_train = torch.from_numpy(data['y_train'].values).float()
        torch_X_val = torch.from_numpy(data['X_val'].values).float()
        torch_y_val = torch.from_numpy(data['y_val'].values).float()
        
        if do_train:
            # train the model
            train(model, torch_X_train, torch_y_train, \
                 torch_X_val, torch_y_val, actionable_indices, experiment_dir, \
                  recourse_loss_weight = w, num_epochs = 15, delta_max = delta_max, lr=lr)
            print("DONE TRAINING")
        
        else:
            model = load_torch_model(weight_dir, weight)

        print("RUNNING EVALUTE")
        run_evaluate(model, data, w, delta_max, actionable_indices, categorical_features, experiment_dir, lam_init = lam_init, \
            data_indices = range(0, 500), thresholds = thresholds_to_eval, max_lam_steps = max_lam_steps, \
            only_eval_at_max_f1 = True)

        print("DONE EVALUATING FOR WEIGHT: ", w)

def run_minority_evaluate(model, dict_data, w, delta_max, actionable_indices, experiment_dir, white_feature_name, \
    prec_targets = [0.65], lam_init = 0.001, max_lam_steps = 10, data_indices = range(0, 500)):
    """
    evaluates the model for recourse/performance metrics on white and minority subsets of data
        uses both the gradient descent and adversarial training algorithms for computing recourse
        outputs different stats for white and minority subsets of data

    :param model: pytorch model to evaluate
    :param dict_data: data in dictionary form
    :param w: lambda value being evaluated
    :param delta_max: parameter defining maximum change in individual feature value
    :param actionable_indices: indices of actionable features
    :param experiment_dir: directory of experiment
    :param white_feature_name: feature name indicating whether a person is white or not
    :param prec_targets: a list of precision targets (we use this to choose thresholds for evaluation)
    :param lam_init: lam_init parameter passed to the gradient descent algorithm
    :param max_lam_steps: max_lam_steps parameter passed to the gradient descent algorithm
    :param data_indices: indices of data subset to evaluate on
    """
    
    model_dir = experiment_dir + str(w) + "/"

    minority_dir = 'minority_exp_fixed_prec/'
    out_dir = model_dir + "test_eval/" + minority_dir 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = dict_data['X_test'].iloc[data_indices]  
    labels = dict_data['y_test'].iloc[data_indices]  

    white_data = data.loc[data[white_feature_name] == 1]
    minority_data = data.loc[data[white_feature_name] == 0]

    white_labels = labels[labels.index.isin(white_data.index)]
    minority_labels = labels[labels.index.isin(minority_data.index)]


    white_tuple = (white_data, white_labels, model_dir + "test_eval/" + minority_dir + "WHITE_wachter_thresholds_test_results.csv", model_dir + "test_eval/" + minority_dir + "WHITE_our_thresholds_test_results.csv")
    minority_tuple = (minority_data, minority_labels, model_dir + "test_eval/" + minority_dir + "MINORITY_wachter_thresholds_test_results.csv", model_dir + "test_eval/" + minority_dir + "MINORITY_our_thresholds_test_results.csv")

    for data, labels, wachter_thresholds_file_name, our_thresholds_file_name in [white_tuple, minority_tuple]:
    
        wachter_thresholds, wachter_precisions, wachter_flipped_proportions, wachter_recourse_proportions, wachter_f1s, wachter_recalls, wachter_accs = [], [], [], [], [], [], []
        our_thresholds, our_precisions, our_flipped_proportions, our_recourse_proportions, our_f1s, our_recalls, our_accs = [], [], [], [], [], [], []


        for prec_target in prec_targets:
            val_data = dict_data['X_val']
            val_labels = dict_data['y_val']
            torch_data = torch.from_numpy(val_data.values).float()
            torch_labels = torch.from_numpy(val_labels.values).float()
            y_prob = model(torch_data).detach().numpy()
            y_true = torch_labels.detach().numpy()

            precisions, recall, all_thresholds = precision_recall_curve(y_true, y_prob)
            idx = (np.abs(precisions - prec_target)).argmin()
            threshold = all_thresholds[idx]
            threshold = round(threshold, 3)

            our_flipped_proportion, our_precision, our_recourse_fraction, our_f1, our_recall, our_acc = our_evaluate(model, white_data, white_labels, w, threshold, delta_max, None, actionable_indices, model_dir, do_print_individual_files = False)
            our_thresholds.append(threshold)
            our_precisions.append(our_precision)
            our_flipped_proportions.append(our_flipped_proportion)
            our_recourse_proportions.append(our_recourse_fraction)
            our_f1s.append(our_f1)
            our_recalls.append(our_recall)
            our_accs.append(our_acc)

            wachter_flipped_proportion, wachter_precision, wachter_recourse_fraction, wachter_f1, wachter_recall, wachter_acc = wachter_evaluate(model, white_data, white_labels, w, threshold, delta_max, lam_init, max_lam_steps, None, actionable_indices, model_dir, do_print_individual_files = False)
            wachter_thresholds.append(threshold)
            wachter_precisions.append(wachter_precision)
            wachter_flipped_proportions.append(wachter_flipped_proportion)
            wachter_recourse_proportions.append(wachter_recourse_fraction)
            wachter_f1s.append(wachter_f1)
            wachter_recalls.append(wachter_recall)
            wachter_accs.append(wachter_acc)

        write_threshold_info(model_dir, w, wachter_thresholds_file_name, wachter_thresholds, wachter_f1s, wachter_accs, wachter_precisions, wachter_recalls, wachter_flipped_proportions, wachter_recourse_proportions)
        write_threshold_info(model_dir, w, our_thresholds_file_name, our_thresholds, our_f1s, our_accs, our_precisions, our_recalls, our_flipped_proportions, our_recourse_proportions)


def wachter_compute_threshold_upperbounds(model, dict_data, weight, delta_max, actionable_indices, epsilons, alpha, model_dir):
    """
    computes threshold upperbounds for PARE guarantees using the gradient descent approximation method of calculating recourse
    evalutes at thresholds in 10 equally spaced increments below that upperbound
    prints results to a file {}_our_threshold_bounds.csv

    :param model: pytorch model to evaluate
    :param dict_data: data in dictionary form (where each value is a dataframe)
    :param weight: weight/lambda value being evaluated (used to create output file name)
    :param delta_max: parameter defining maximum change in individual feature value
    :param actionable_indices: indices of actionable features
    :param epsilons: epsilon values to evaluate at
    :param alpha: the probability that the guarantee holds
    :model_dir: model (weight) specific directory within experiment directory
    """

    test_eval_dir = model_dir + "test_eval/"

    if not os.path.exists(test_eval_dir):
        os.makedirs(test_eval_dir)

    file_name = test_eval_dir + str(weight) + '_our_threshold_bounds.csv'

    model.eval()


    X_val, y_val = dict_data['X_val'], dict_data['y_val']
    data = X_val  
    labels = y_val

    torch_data = torch.from_numpy(data.values).float()
    torch_labels = torch.from_numpy(labels.values)

    probs = []

    logger = logging.getLogger('alibi.explainers.counterfactual')
    logging.basicConfig(level=logging.CRITICAL)

    for i in range(len(torch_labels)):

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
                assert(action <= delta_max).all()

            else:
                num_no_recourses += 1
                num_none_returned += 1
        except UnboundLocalError as e:
            num_no_recourses += 1
            if do_print:
                print(e)
                print("no success")
        except AssertionError as e:
            print("assertion error")
        num_neg_instances += 1

        probs.append(recourse.cf['proba'])

    probs = np.array(probs)

    threshold_bounds = []
    ds = []
    test_f1s = []
    test_accs = []
    test_precisions = []
    test_recalls = []
    max_thresh = []
    test_recourse_proportions = []

    for epsilon in epsilons:
        t = compute_t(probs, 1 - epsilon, alpha)
        threshold_bounds.append(t)
        ds.append(d)

        max_f1 = 0
        max_t = None
        for t_cand in np.linspace(0, t, 10):

            y_pred = [0.0 if a < t_cand else 1.0 for a in (model(torch_data).detach().numpy())]
            y_true = ((torch_labels).detach().numpy())
            y_prob_pred = model(torch_data).detach().numpy()
        
            f1 = round(f1_score(y_true, y_pred), 3)     

            if f1 > max_f1:
                max_f1 = f1
                max_t = round(t_cand, 3)

        test_torch_data = torch.from_numpy(dict_data['X_test'].values).float()
        test_labels = torch.from_numpy(dict_data['y_test'].values).float()

        flipped = 0
        for i in range(len(test_labels)):
            x = test_torch_data[i]              # data point
            
            if model(x).item() < max_t:
                delta_opt = calc_delta_opt(model, x, delta_max, actionable_indices)
                if (model(x + delta_opt).item()) >= max_t:
                    flipped += 1

        y_pred = [0.0 if a < max_t else 1.0 for a in (model(test_torch_data).detach().numpy())]
        y_true = ((test_labels).detach().numpy())
        num_pos = sum(y_pred)
        test_f1 = round(f1_score(y_true, y_pred), 3)  
        test_acc = round(np.sum(y_pred == y_true)/(y_true).shape[0], 3)
        test_prec = round(precision_score(y_true, y_pred), 3)  
        test_recall = round(recall_score(y_true, y_pred), 3)  
        test_recourse_proportion = round((num_pos + flipped)/len(test_labels), 3)

        max_thresh.append(max_t)
        test_f1s.append(test_f1)
        test_accs.append(test_acc)
        test_precisions.append(test_prec)
        test_recalls.append(test_recall)
        test_recourse_proportions.append(test_recourse_proportion)



    thresholds_data = {}
    thresholds_data['threshold_bounds'] = threshold_bounds
    thresholds_data['epsilons'] = epsilons
    thresholds_data['d'] = ds
    thresholds_data['test_f1s'] = test_f1s
    thresholds_data['test_accs'] = test_accs
    thresholds_data['test_precisions'] = test_precisions
    thresholds_data['test_recalls'] = test_recalls
    thresholds_data['test_recourse_proportions'] = test_recourse_proportions


    thresholds_data['actual_thresh'] = max_thresh

    thresholds_df = pd.DataFrame(data=thresholds_data)
    thresholds_df['threshold_bounds'] = thresholds_df['threshold_bounds'].round(3)

    print(threshold_bounds)

    thresholds_df.to_csv(file_name, index_label='index')