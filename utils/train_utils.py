import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import itertools
from scipy.optimize import linprog
from torch.autograd import grad
from sklearn import metrics
from tqdm import tqdm
import time
import numpy as np
import os
import pandas as pd

def calc_delta_opt(model, x, delta_max, actionable_indices):
    """
    calculate the optimal delta using linear program

    :param model: pytorch model
    :param delta_max: parameter defining maximum change in individual feature value
    :actionable_indices: indices of actionable features

    :returns: torch tensor with optimal delta value
    """

    loss_fn = torch.nn.BCELoss()
                            
    A_eq = np.empty((0,len(x)), float)
    
    b_eq = []
    
    for idx in range(len(x)):
        if idx not in actionable_indices:
            A_temp = np.zeros((1,len(x)))
            A_temp[0, idx] = 1.0
            A_eq = np.append(A_eq, np.array(A_temp), axis=0)
            b_eq.append(0.0)
    
    b_eq = np.array(b_eq)
    
    x.requires_grad = True
    x_loss = loss_fn(model(x), torch.tensor([1.0]).float())
    gradient_x_loss = grad(x_loss, x)[0]
    
    c = list(np.array(gradient_x_loss) * np.array([-1]*len(gradient_x_loss)))
    bound = (-delta_max, delta_max)
    bounds = [bound]*len(gradient_x_loss)
    
    res = linprog(c, bounds=bounds, A_eq = A_eq, b_eq = b_eq, method='simplex')
    delta_opt = res.x # the delta value that maximizes the function

    return torch.tensor(delta_opt).float()

def combined_loss(model, output, target, delta, x, loss_fn, recourse_loss_weight = 1.0):
    """
    calculate the combined loss function for training

    :param model: pytorch model
    :param output: model prediction
    :param target: label
    :param delta: optimal delta found
    :param x: input
    :param loss_fn: base loss function used for training    

    :returns: calculated combined loss value
    """

    bce_loss = torch.nn.BCELoss()
    loss = bce_loss(model(x + delta), torch.tensor([1.0]).float())
    return recourse_loss_weight * loss + loss_fn(output, target.reshape(1))

def write_epoch_train_info(train_file_name, y_val, epoch_start, maj_label, min_label, n):
    """
    writes epoch-specific training information to training log

    :param train_file_name: name of training log file
    :param y_val: labels for val data
    :param epoch_start: time when epoch started
    :param maj_label: majority label (0 or 1)
    :param min_label: minority label (0 or 1)
    :param n: num epochs

    """
    maj_baseline_preds = (np.full(y_val.shape, maj_label)).ravel().tolist()
    min_baseline_preds = (np.full(y_val.shape, min_label)).ravel().tolist()

    training_file = open(train_file_name, "a")
    training_file.write("EPOCH: " + str(n) + "\n")
    training_file.write("time for epoch: " + str(round((time.time() - epoch_start)/60, 3)) + " minutes" + "\n")
    training_file.write("val maj ({}) baseline accuracy: {}\n".format(maj_label, round(np.sum(np.full(y_val.shape, maj_label) == y_val)/(y_val).shape[0], 3)))
    training_file.write("val min ({}) baseline accuracy: {}\n".format(min_label, round(np.sum(np.full(y_val.shape, min_label) == y_val)/(y_val).shape[0], 3)))
    training_file.write("val min baseline f1: {}\n".format(round(f1_score((y_val).ravel().tolist(), min_baseline_preds), 3)))        
    training_file.write("val maj baseline f1: {}\n\n".format(round(f1_score((y_val).ravel().tolist(), maj_baseline_preds), 3)))        
    training_file.close()

def write_best_val_model(best_model_stats_file_name, best_model_name, lr, n, delta_max, model):
    """
    writes params for model with best val loss
    saves model to file

    :param best_model_stats_file_name: name of best model stats file
    :param best_model_name: name of file with best model parameters
    :param lr: learning rate
    :param n: num epochs
    :param delta_max: delta_max parameter
    :param model: model to save

    """
    text_file = open(best_model_stats_file_name, "w")
    text_file.write("lr: " + str(lr) + "\n")
    text_file.write("num epochs: " + str(n) + "\n")
    text_file.write("delta max: " + str(delta_max) + "\n\n")
    text_file.close()   
    torch.save(model, best_model_name)

def write_stats_at_threshold(train_file_name, best_model_stats_file_name, model, X_train, X_val, y_train, y_val, \
    t, val_flipped, best_epoch, num_negative, num_positive):
    """
    writes stats for model at threshold to two different files (with names train_file_name & best_model_stats_file_name)

    :param train_file_name: name of training log file
    :param best_model_stats_file_name: name of best model stats file
    :param model: model
    :param X_train: train X data
    :param X_val: val X data
    :param y_val: val y data
    :param t: threshold
    :param val_flipped: number of val instances for which adding calculated optimal delta led to a flip in prediction (i.e. recourse neg)
    :param best_epoch: Boolean for whether this is epoch with best val loss
    :param num_negative: number of negative val preds for this epoch
    :param num_positive: number of positive val preds for this epoch

    :return: tuple (val_precision, val_recourse_proportion, val_flipped_proportion, val_f1)
        val_precision: precision at threshold t
        val_recourse_proportion: portion of instances with recourse at threshold t
        val_flipped_proportion: portion of neg instances with recourse at threshold t
        val f1: f1 score at threshold t
    """
    
    # compute stats at this threshold
    val_y_pred = np.array([0.0 if a < t else 1.0 for a in (model(X_val).detach().numpy())])
    val_y_true = (y_val).detach().numpy()

    val_acc = np.sum(val_y_pred == val_y_true)/val_y_true.shape[0]
    val_f1 = round(f1_score(y_val, val_y_pred), 3)
    val_precision = precision_score(y_val, val_y_pred)
    val_recall = recall_score(y_val, val_y_pred)
        
    train_preds = [0.0 if a < t else 1.0 for a in model(X_train).detach().numpy().ravel()]
    if num_negative != 0:
        val_proportion_flipped = round(val_flipped/num_negative, 3)
    else:
        val_proportion_flipped = 0

    training_file = open(train_file_name, "a")
    training_file.write("\nSTATS FOR threshold = " + str(t) + ":\n")
    training_file.write("val preds: " + str(np.unique(val_y_pred, return_counts = True)) + "\n")
    training_file.write("val accuracy: {}\n".format(round(val_acc, 3)))
    training_file.write("val f1: {}\n".format(round(val_f1, 3)))
    training_file.write("val precision: {}\n".format(round(val_precision, 3)))
    training_file.write("val recall: {}\n".format(round(val_recall, 3)))
    training_file.write("val num flipped: {}; {}/{}\n".format(val_proportion_flipped, val_flipped, num_negative))
    training_file.write("val num with recourse: {}; {}/{}\n".format(round((val_flipped + num_positive)/len(y_val), 3), (val_flipped + num_positive), len(y_val)))
    training_file.write("train accuracy: {}\n".format(round(np.sum((train_preds \
                 == ((y_train).detach().numpy()))/((y_train).detach().numpy()).shape[0]), 3)))
    training_file.close()

    val_recourse_proportion = round((val_flipped + num_positive)/len(y_val), 3)

    if best_epoch:
    
        text_file = open(best_model_stats_file_name, "a")
        text_file.write("THRESHOLD: " + str(t) + "\n")
        text_file.write("val preds: " + str(np.unique(val_y_pred, return_counts = True)) + "\n")
        text_file.write("VAL PRECISION: " + str(round(val_precision, 3)) + "\n")
        text_file.write("val accuracy: " + str(round(val_acc, 3)) + "\n")
        text_file.write("val f1: " + str(round(val_f1, 3)) + "\n")
        text_file.write("val precision: " + str(round(val_precision, 3)) + "\n")
        text_file.write("val recall: " + str(round(val_recall, 3)) + "\n")
        text_file.write("val num flipped: {}; {}/{}\n".format(val_proportion_flipped, val_flipped, num_negative))
        text_file.write("val num with recourse: {}; {}/{}\n".format(val_recourse_proportion, (val_flipped + num_positive), len(y_val)))
        text_file.write("-------------------\n\n")
        text_file.close()

    return val_precision, val_recourse_proportion, val_proportion_flipped, val_f1

def train(model, X_train, y_train, X_val, y_val, actionable_indices, experiment_dir, \
          num_epochs = 12, delta_max = 0.75, batch_size = 10, lr = 0.002, \
          recourse_loss_weight=1):
    """
    trains model and writes training stats to file ({}_model_training_info.txt)
    also evaluates on validation data after each epoch for thresholds from [0.0, 1.0] in increments of 0.05 and: 
        saves model with best val loss
        writes val stats to val log files ({}_best_model_val_info.txt. {}_val_thresholds_info.csv)

    :param model: model
    :param X_train: train X data
    :param y_train: train y data
    :param X_val: val X data
    :param y_val: val y data
    :param actionable_indices: indices of actionable features
    :param experiment_dir: directory where experiment is stored
    :param num_epochs: number of epochs
    :param delta_max: maximum amount a feature can be changed by in calculating recourse
    :param batch_size: number of instances after which to make the gradient update
    :param lr: learning rate
    :param recourse_loss_weight: lambda value (weights the recourse/adversarial part of our training objective)

    :return: tuple (val_precision, val_recourse_proportion, val_flipped_proportion, val_f1)
        val_precision: precision at threshold t
        val_recourse_proportion: portion of instances with recourse at threshold t
        val_flipped_proportion: portion of neg instances with recourse at threshold t
        val f1: f1 score at threshold t
    """   
    
    weight_dir = experiment_dir + str(recourse_loss_weight) + '/'

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # train_file_name: name of train log file
    train_file_name = weight_dir + str(recourse_loss_weight) + "_model_training_info.txt"
    best_model_stats_file_name = weight_dir + str(recourse_loss_weight) + "_best_model_val_info.txt"
    best_model_name = weight_dir + str(recourse_loss_weight) + '_best_model.pt'
    best_model_thresholds_file_name = weight_dir + str(recourse_loss_weight) + '_val_thresholds_info.csv'

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # count the +/- labels to compute weight for minority class
    pos_labels = torch.sum(y_train).item()
    neg_labels = (len(y_train) - pos_labels)

    if pos_labels > neg_labels:
        minority_class_weight = pos_labels/neg_labels
        min_label = 0.0
        maj_label = 1.0
    else:
        min_label = 1.0
        maj_label = 0.0
        minority_class_weight = neg_labels/pos_labels    

    # open training log file: 'weight_dir/WEIGHT_model_training_info.txt'
    # write parameters
    training_file = open(train_file_name, "w")
    training_file.write("len(train): " + str(len(y_train)) + "\n")
    training_file.write("train # pos: " + str(pos_labels) + "\n")
    training_file.write("train # neg: " + str(neg_labels) + "\n")
    training_file.write("len(val): " + str(len(y_val)) + "\n\n")
    
    training_file.write("lr: " + str(lr)+ "\n")
    training_file.write("num epochs: " + str(num_epochs) + "\n")
    training_file.write("delta max: " + str(delta_max) + "\n")
    training_file.write("recourse weight in loss function: " + str(recourse_loss_weight) + "\n\n")
    
    training_file.close()


    # set default values
    loss = 0
    best_val_loss = 10000000    
    best_thresholds = None


    for n in range(num_epochs):

        epoch_start = time.time()
        train_epoch_loss = 0

        print("STARTING epoch: ", n)
    

        for i in tqdm(range(len(y_train))):
        
            label = y_train[i]
            x = X_train[i]
            y_pred = model(x)
            
            # define loss function, upweight instance if minority class
            weight = torch.tensor([minority_class_weight]) if (label == min_label) else torch.tensor([1.0])
            assert ((label == min_label and weight.item() > 1.0) or (label == maj_label and weight.item() == 1.0))
            loss_fn = torch.nn.BCELoss(weight=weight)
            
            # calculate the weighted combined loss
            delta_opt = calc_delta_opt(model, x, delta_max, actionable_indices)
            loss += combined_loss(model, y_pred, label, delta_opt, x, loss_fn, recourse_loss_weight=recourse_loss_weight)
               

            # take step in direction of gradient every (batch_size) # of instances
            # reset loss to 0
            if i % batch_size == 0:    
                optimizer.zero_grad()
                loss = loss/(batch_size)
                loss.backward()

                train_epoch_loss += loss.item()
                
                optimizer.step()

                loss = 0
        
        # VAL EVALUATION
        print("TRAIN LOSS FOR EPOCH: ", round(train_epoch_loss, 3))
        model.eval()

        # val loss for epoch
        epoch_val_loss = 0
        
        print("EVALUATING ON VAL DATA")

        # y_true: (np.array) true labels for validation data
        # y_true_torch = (torch tensor) of true labels for validation data
        # y_prob_pred: (np.array) model probability predictions for validation data
        y_true = ((y_val).detach().numpy())
        y_true_torch = y_val
        y_prob_pred = model(X_val).detach().numpy()



        # get metrics
        precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_prob_pred, pos_label=1.0)

        # add custom thresholds
        thresholds = ([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

        # FOR VAL
        flipped_epoch_by_threshold = [0 for a in thresholds]
        negative_epoch_by_threshold = [0 for a in thresholds]     
        
        # iterate through validation data:

        for i in range(len(y_true_torch)):

            # define loss function, upweight instance if minority class
            weight = torch.tensor([minority_class_weight]) if (label == min_label) else torch.tensor([1.0])
            loss_fn = torch.nn.BCELoss(weight=weight)
                
            x = X_val[i]              # data point
            label = y_val[i]
            y_pred = model(x)

            # calculate the weighted combined loss
            delta_opt = calc_delta_opt(model, x, delta_max, actionable_indices)

            # for each threshold, keep track of negative predictions and flipped predictions
            for t_idx, t in enumerate(thresholds):
                if y_pred.item() < t:
                    negative_epoch_by_threshold[t_idx] += 1
                    if model(x + delta_opt).detach().numpy() >= t:
                        flipped_epoch_by_threshold[t_idx] += 1  

            loss += combined_loss(model, y_pred, label, delta_opt, x, loss_fn, recourse_loss_weight=recourse_loss_weight)

            # add average batch loss to epoch_val_loss (val loss for epoch)
            if i % batch_size == 0:    
                loss = loss/(batch_size)                
                epoch_val_loss += loss
                loss = 0

        print("VAL LOSS FOR EPOCH: ", round(epoch_val_loss.item(), 3))

        # write training stats for epoch
        write_epoch_train_info(train_file_name, y_true, epoch_start, maj_label, min_label, n)

        # determine if this is the model with the best val loss
        # if so, save
        if epoch_val_loss < best_val_loss:
            best_epoch = True
            best_val_loss = epoch_val_loss

            write_best_val_model(best_model_stats_file_name, best_model_name, lr, n, delta_max, model)

            best_thresholds = thresholds
            
        else:
            best_epoch = False

        # all of these are validation and are ordered by thresholds
        precision_by_threshold = []
        recourse_proportion_by_threshold = []
        flipped_proportion_by_threshold = []
        f1_by_threshold = []

        # for each threshold, compute and record stats
        for t_idx, t in enumerate(best_thresholds):

            val_flipped = flipped_epoch_by_threshold[t_idx]
            num_negative = negative_epoch_by_threshold[t_idx]
            num_positive = len(y_true) - num_negative
            val_precision, val_recourse_proportion, val_flipped_proportion, val_f1 = write_stats_at_threshold(train_file_name, best_model_stats_file_name, model, X_train, X_val, y_train, y_val, \
                t, val_flipped, best_epoch, num_negative, num_positive)

            precision_by_threshold.append(round(val_precision, 3))
            recourse_proportion_by_threshold.append(val_recourse_proportion)
            flipped_proportion_by_threshold.append(val_flipped_proportion)
            f1_by_threshold.append(val_f1)

        # if the best epoch, write information about thresholds and various metrics
        if best_epoch:
            thresholds_data = {}

            print(best_thresholds)

            thresholds_data['thresholds'] = best_thresholds
            thresholds_data['precisions'] = precision_by_threshold
            thresholds_data['flipped_proportion'] = flipped_proportion_by_threshold
            thresholds_data['recourse_proportion'] = recourse_proportion_by_threshold
            thresholds_data['f1s'] = f1_by_threshold

            thresholds_df = pd.DataFrame(data=thresholds_data)
            thresholds_df['thresholds'] = thresholds_df['thresholds'].round(3)
            thresholds_df.to_csv(best_model_thresholds_file_name, index_label='index')
                
        training_file = open(train_file_name, "a")
        training_file.write("-------------------\n\n")
        training_file.close()
             
        model.train()    
