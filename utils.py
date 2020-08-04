from alibi.explainers import CounterFactual
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import torch
import numpy as np
import pandas as pd
from train_utils import *
from small_model import *
from big_model import *

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

def wachter_evaluate(model, X_test, y_test, weight, threshold, delta_max, lam_init, data_indices, actionable_indices, model_dir):
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

    print_test_results(file_name, model, threshold, weight, data, labels, precision)

    f = open(file_name, "a")
    print("LAM INIT: {}".format(lam_init), file=f)
    print("DELTA MAX: {}".format(delta_max), file=f)
    print("len(instances) evaluated on: {}\n\n".format(len(data_indices)), file=f)
    f.close()

    # consider all lams from lam_init/10 to lam_init; 
    #if cannot find any counterfactual instances using those values, i am assuming it is an unsolvable problem

    do_print = False

    scores = (pred_function(model, data.values)[:, 0])

    neg_test_preds = np.nonzero(np.array(scores) < threshold)[0]

    num_no_recourses = 0
    num_neg_instances = 0
    num_none_returned = 0

    for i in tqdm(neg_test_preds, total = len(neg_test_preds)):
        sample = data.iloc[i].values.reshape(1,-1)
        mins = sample[0].copy()
        maxs = sample[0].copy()
        for ai in actionable_indices:
            mins[ai] = mins[ai] - delta_max
            maxs[ai] = maxs[ai] + delta_max
        mins = mins.reshape(1,-1)
        maxs = maxs.reshape(1,-1)
        tf.reset_default_graph()
        explainer = CounterFactual(lambda x: pred_function(model, x), \
                               shape=(1,) + data.iloc[0].values.shape, \
                               tol=(1.0 - threshold), target_class='other', \
                               feature_range = (mins, maxs), lam_init = lam_init)
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
    
    f = open(file_name, "a")
    print("num none returned: {}/{}, {}".format(num_none_returned, num_neg_instances, none_returned_proportion), file=f)
    print("flipped: {}/{}, {}".format((num_neg_instances - num_no_recourses), num_neg_instances, flipped_proportion), file=f)
    print("proportion with recourse: {}".format(recourse_fraction), file=f)
    print("--------\n\n", file=f) 
    f.close()

    return flipped_proportion, precision, recourse_fraction


def our_evaluate(model, X_test, y_test, weight, threshold, delta_max, actionable_indices, model_dir):
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

    torch_data = torch.from_numpy(X_test.values).float()
    torch_labels = torch.from_numpy(y_test.values)

    # this part is redundant with print_test_results
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    precision = round(precision_score(y_true, y_pred), 3)
    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds

    print_test_results(file_name, model, threshold, weight, X_test, y_test, precision)

    f = open(file_name, "a")
    print("DELTA MAX: {}".format(delta_max), file=f)

    loss_fn = torch.nn.BCELoss()

    negative_instances, flipped = 0, 0
    total_instances = len(y_test)

    for i in range(len(torch_labels)):
        x = torch_data[i]              # data point
        y_pred = model(x).item()
        
        if y_pred < threshold: #negative pred
            negative_instances += 1
            delta_opt = calc_delta_opt(model, x, delta_max, actionable_indices)
            if model(x + delta_opt) > threshold:
                flipped += 1

    recourse_fraction = round((flipped + pos_preds)/total_instances, 3)
    f.write("\nlen(test): {}".format(len(y_test)))
    if negative_instances != 0:
        f.write("\nflipped/negative: {}, {}/{}".format(round(flipped/negative_instances, 3), flipped, negative_instances))
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

    return flipped, precision, recourse_fraction  
        
def get_threshold_info(model_dir, weight):
    # this is based on the value of this name in the train function
    best_model_thresholds_file_name = model_dir + str(weight) + '_val_thresholds_info.csv'
    threshold_df = pd.read_csv(best_model_thresholds_file_name, dtype=np.float64, index_col = 'index')
    return threshold_df

def write_threshold_info(model_dir, weight, thresholds_file_name, thresholds, precisions, flipped_proportion, recourse_proportion):
    thresholds_data = {}
    thresholds_data['thresholds'] = thresholds
    thresholds_data['precisions'] = precisions
    thresholds_data['flipped_proportion'] = flipped_proportion
    thresholds_data['recourse_proportion'] = recourse_proportion

    thresholds_df = pd.DataFrame(data=thresholds_data)
    thresholds_df['thresholds'] = thresholds_df['thresholds'].round(3)
    thresholds_df.to_csv(thresholds_file_name, index_label='index')

def run_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, \
    thresholds = None, lam_init = 0.01, data_indices = range(0, 500)):
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
    wachter_thresholds, wachter_precisions, wachter_flipped_proportions, wachter_recourse_proportions = [], [], [], []
    our_thresholds, our_precisions, our_flipped_proportions, our_recourse_proportions = [], [], [], []

    # thresholds arg not supplied, read in thresholds from validation evaluation during training and use those
    if thresholds == None:
        threshold_df = get_threshold_info(model_dir, w)
        thresholds = list(threshold_df['thresholds'])

    print("THRESHOLDS: ", thresholds)
    for threshold in thresholds:
        threshold = round(threshold, 3)
        print("THR: ", threshold)
        our_flipped_proportion, our_precision, our_recourse_fraction = our_evaluate(model, data['X_test'], data['y_test'], w, threshold, delta_max, actionable_indices, model_dir)
        our_thresholds.append(threshold)
        our_precisions.append(our_precision)
        our_flipped_proportions.append(our_flipped_proportion)
        our_recourse_proportions.append(our_recourse_fraction)

        wachter_flipped_proportion, wachter_precision, wachter_recourse_fraction = wachter_evaluate(model, data['X_test'], data['y_test'], w, threshold, delta_max, lam_init, data_indices, actionable_indices, model_dir)
        wachter_thresholds.append(threshold)
        wachter_precisions.append(wachter_precision)
        wachter_flipped_proportions.append(wachter_flipped_proportion)
        wachter_recourse_proportions.append(wachter_recourse_fraction)

    write_threshold_info(model_dir, w, wachter_thresholds_file_name, wachter_thresholds, wachter_precisions, wachter_flipped_proportions, wachter_recourse_proportions)
    write_threshold_info(model_dir, w, our_thresholds_file_name, our_thresholds, our_precisions, our_flipped_proportions, our_recourse_proportions)
                    

def run(data, actionable_indices, experiment_dir, weights):
    
    lr = 0.002 # changed this for compas training
    delta_max = 0.75
    fixed_precisions = [0.4, 0.5, 0.6, 0.7]

    for w in weights:
        print("WEIGHT: ", w)
        model = SmallModel(len(data['X_train'].values[0]))
        
        torch_X_train = torch.from_numpy(data['X_train'].values).float()
        torch_y_train = torch.from_numpy(data['y_train'].values).float()
        torch_X_val = torch.from_numpy(data['X_val'].values).float()
        torch_y_val = torch.from_numpy(data['y_val'].values).float()
        
        # train the model
        train(model, torch_X_train, torch_y_train, \
             torch_X_val, torch_y_val, actionable_indices, experiment_dir, \
              recourse_loss_weight = w, num_epochs = 30, delta_max = delta_max, lr=lr, \
              fixed_precisions = fixed_precisions)


        run_evaluate(model, data, w, delta_max, actionable_indices, experiment_dir, lam_init = 0.01)
