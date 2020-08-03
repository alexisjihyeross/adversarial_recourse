from alibi.explainers import CounterFactual
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import torch
import numpy as np

def pred_function(model, np_x):
    torch_x = torch.from_numpy(np_x).float()
    model_pred = model(torch_x).detach().numpy()
    probs = np.concatenate((model_pred, 1-model_pred), axis=1)
    return probs

def wachter_evaluate(model, X_test, y_test, weight, threshold, delta_max, lam_init, data_indices, actionable_indices, model_dir):
    """
    calculate the optimal delta using linear program

    :param model: pytorch model to evaluate
    :param X_test: X test data (e.g. adult_data['X_test'])
    :param y_test: y test data (e.g. adult_data['y_test'])
    :param weight: weight being evaluated (used to name file)
    :param threshold: threshold to use in evaluation
    :param delta_max: parameter defining maximum change in individual feature value
    :param lam_init: lam hyperparameter for wachter evaluation
    :data_indices: subset of data to evaluate on (e.g. something like range(100, 300))
    :actionable_indices: indices of actionable features
    :model_dir: model (weight) specific directory within experiment directory

    :returns: torch tensor with optimal delta value
        this optimal value always has either -/+ delta_max for each actionable index
    """

    wachter_dir = model_dir + "wachter/"

    if not os.path.exists(wachter_dir):
        os.makedirs(wachter_dir)

    file_name = wachter_dir + str(weight) + '_wachter_test.txt'

    model.eval()
    data = X_test.iloc[data_indices]
    torch_data = torch.from_numpy(data.values).float()
    
    labels = y_test.iloc[data_indices]
    torch_labels = torch.from_numpy(labels.values)
    
    # evaluate on subset of data:
    y_pred = [0.0 if a < threshold else 1.0 for a in (model(torch_data).detach().numpy())]
    y_true = ((torch_labels).detach().numpy())
    y_prob_pred = model(torch_data).detach().numpy()
    
    f1 = round(f1_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred), 3)
    recall = round(recall_score(y_true, y_pred), 3)
    acc = round(np.sum(y_pred == y_true)/(y_true).shape[0], 3)

    pos_preds = np.sum(y_pred)
    neg_preds = len(y_pred) - pos_preds
    
    f = open(file_name, "w")
    
    print("WEIGHT: {}".format(weight), file=f)
    print("THRESHOLD: {}".format(threshold), file=f)
    print("LAM INIT: {}".format(lam_init), file=f)
    print("len(instances) evaluated on:: {}\n\n".format(len(data_indices)), file=f)
    print("performance results on subset of test data:")
    print("pos preds: ", pos_preds, file=f)
    print("f1: ", f1, file=f)
    print("precision: ", precision, file=f)
    print("recall: ", recall, file=f)
    print("acc: ", acc, file=f)

    # consider all lams from lam_init/10 to lam_init; 
    #if cannot find any counterfactual instances using those values, i am assuming it is an unsolvable problem

    do_print = False

    scores = (pred_function(model, data.values)[:, 0])

    neg_test_preds = np.nonzero(np.array(scores) < threshold)[0]

    num_no_recourses = 0
    num_neg_instances = 0
    num_none_returned = 0

    for i in tqdm(neg_test_preds):
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

    flipped_proportion = round((num_neg_instances-num_no_recourses)/num_neg_instances, 3)
    recourse_fraction = round((pos_preds + num_with_recourses)/len(y_pred), 3)
    
    assert(num_neg_instances == neg_preds)
    assert((pos_preds + num_neg_instances) == len(y_pred))
    
    print("\nDELTA MAX: ", delta_max, file=f)
    print("LAM INIT: ", lam_init, file=f)
    print("num none returned: {}/{}, {}".format(num_none_returned, num_neg_instances, round(num_none_returned/num_neg_instances, 3)), file=f)
    print("flipped: {}/{}, {}".format((num_neg_instances - num_no_recourses), num_neg_instances, flipped_proportion), file=f)
    print("proportion with recourse: {}".format(recourse_fraction), file=f)
    print("--------\n\n", file=f) 
    f.close()

    return flipped_proportion, precision, recourse_fraction

                    