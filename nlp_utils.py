import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW

from nltk.tree import Tree
from nltk.tokenize.treebank import TreebankWordDetokenizer

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from sklearn.metrics import f1_score, precision_score, recall_score

import pandas as pd
import time

def get_threshold_info(model_dir, weight):
    # this is based on the value of this name in the train function
    best_model_thresholds_file_name = model_dir + str(weight) + '_val_thresholds_info.csv'
    threshold_df = pd.read_csv(best_model_thresholds_file_name, dtype=np.float64, index_col = 'index')
    return threshold_df

def load_model(device, model_name = 'bert-base-uncased'):
    print("loading model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    model.train()

    tokenizer = BertTokenizer.from_pretrained(model_name)

    print("done.")
    return model, tokenizer

def load_trained_model(weight_dir, weight):
    model_name = weight_dir + str(weight) + "_best_model.pt"
    model = torch.load(model_name)
    model.eval()
    return model

def get_sst_data(file_path):
    with open(file_path, "r") as data_file:
        lines = list(data_file.readlines())
        texts = []
        labels = []
        for i, line in enumerate(lines):
            line = line.strip("\n")
            if not line:
                continue
            parsed_line = Tree.fromstring(line)
            text = (TreebankWordDetokenizer().detokenize(parsed_line.leaves()))
            sentiment = int(parsed_line.label())
            if sentiment < 2:
                sentiment = 0.0
            elif sentiment == 2:
                continue
            else:
                sentiment = 1.0
            texts.append(text)
            labels.append(sentiment)
    return texts, labels

def antonyms(term):
    max_tries = 100
    counter = 0
    while counter < max_tries:
        try:
            response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
            soup = BeautifulSoup(response.text, 'lxml')
            return [span.text for span in soup.findAll('a', {'class': 'css-itvgb'})]
        except:
            counter += 1
            continue
    return None

def get_candidates(model, text, max_candidates):
    words = word_tokenize(text)
    candidates = [None] * max_candidates
    counter = 0
    for word in words:
        if wn.synsets(word) == []:
            continue
        tmp = wn.synsets(word)[0].pos()
        # if not adjective or noun, continue
        if tmp != "a" and tmp != "n":
            continue
        for a in antonyms(word):
            candidates[counter] = (TreebankWordDetokenizer().detokenize([a.rstrip() if x == word else x for x in words]))
            counter += 1
            if counter >= max_candidates:
                return list(filter(None.__ne__, candidates))
    return list(filter(None.__ne__, candidates))

def get_delta_opt(model, tokenizer, device, text, max_candidates):
    cands = get_candidates(model, text, max_candidates)
    max_prob = 0
    found_cand = False
    for c in cands:
        input_ids, labels = get_tensors(device, tokenizer, c, 1.0)
        cand_logits, cand_labels, cand_prob = get_pred(model, tokenizer, device, input_ids, labels)
        if cand_prob > max_prob:
            max_cand = c
            max_prob = cand_prob
            max_logits = cand_logits
            max_prob = cand_prob
            found_cand = True
            del input_ids
            del labels
        else:
            del cand_logits
            del cand_labels
            del cand_prob
            torch.cuda.empty_cache()
    if not found_cand:
        input_ids, labels = get_tensors(device, tokenizer, text, 1.0)
        max_logits, max_labels, max_prob = get_pred(model, tokenizer, device, input_ids, labels)
        max_cand = text
        del input_ids
        del labels
    return max_cand, max_logits, max_prob

def get_pred(model, tokenizer, device, input_ids, labels):
    outputs = model(input_ids)
    logits = outputs[0]
    pos_prob = torch.nn.Softmax(dim=-1)(logits)[:, -1]
    return logits, labels, pos_prob

def get_tensors(device, tokenizer, text, label):
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids']
    input_ids = encoding.to(device)
    labels = torch.LongTensor([label]).to(device)
    return input_ids, labels

def train_nlp(model, tokenizer, weight_dir, thresholds_to_eval, recourse_loss_weight, max_candidates = 10):

    training_file_name = weight_dir + str(recourse_loss_weight) + "_model_training_info.txt"

    training_file = open(training_file_name, "w")

    # get data
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load model and tokenizer    
    train_texts, train_labels = get_sst_data('data/nlp_data/train.txt')
    #train_texts = train_texts[0:50]
    #train_labels = train_labels[0:50]


    dev_texts, dev_labels = get_sst_data('data/nlp_data/dev.txt')
    #dev_texts = dev_texts[0:50]
    #dev_labels = dev_labels[0:50]

    batch_size = 1 

    lr = 2e-5
    num_warmup_steps = 0
    num_epochs = 2
    num_train_steps = len(train_texts)/batch_size * num_epochs


    print("len(train): ", len(train_texts), file = training_file)
    print("train # pos: ", np.sum(np.array(train_labels)), file = training_file)
    print("train # neg: ", len(train_texts) - np.sum(np.array(train_labels)), file = training_file)
    print("len(val): ", len(dev_texts), file = training_file)

    print("", file = training_file)

    print("lr: ", lr, file = training_file)
    print("num warmup steps: ", num_warmup_steps, file = training_file)
    print("num epochs: ", num_epochs, file = training_file)
    print("max candidates: ", max_candidates, file = training_file)

    optim = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps, num_train_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    def combined_loss(model, device, logits, labels, delta_logits, loss_fn, recourse_loss_weight):
        normal_loss = loss_fn(logits, labels)
        recourse_loss = loss_fn(delta_logits, torch.LongTensor([1.0]).to(device))
        return recourse_loss * recourse_loss_weight + normal_loss

    best_val_loss = 100000000

    flipped_by_thresh = {thresh: 0 for thresh in thresholds_to_eval}
    negative_by_thresh = {thresh: 0 for thresh in thresholds_to_eval}

    for epoch in range(num_epochs):
        batch_loss, train_epoch_loss, train_correct = 0, 0, 0
        
        print("EPOCH: ", epoch)
        epoch_start = time.time()
        model.train()
        
        pos_probs = []
        num_no_cands = 0

        for i, (text, label) in tqdm(enumerate(zip(train_texts, train_labels)), total = len(train_texts)):
            input_ids, labels = get_tensors(device, tokenizer, text, label)
            logits, labels, pos_prob = get_pred(model, tokenizer, device, input_ids, labels)
            cand_text, delta_logits, delta_prob = get_delta_opt(model, tokenizer, device, text, max_candidates)
            if cand_text == text:
                num_no_cands += 1
            batch_loss += combined_loss(model, device, logits, labels, delta_logits, loss_fn, recourse_loss_weight)
                
                
            if i % batch_size == 0:    
                model.zero_grad() 
                batch_loss.backward()
                train_epoch_loss += batch_loss.item()
                optim.step()
                scheduler.step()
                del batch_loss
                torch.cuda.empty_cache()
                batch_loss = 0
            
            pos_probs.append(pos_prob.item())

            for t in thresholds_to_eval:
                if pos_prob.item() < t:
                    negative_by_thresh[t] += 1
                    if delta_prob.item() >= t:
                        flipped_by_thresh[t] += 1

        # write train info
        print("----------", file = training_file)
        print("", file = training_file)
        print("EPOCH: ", epoch, file = training_file)
        print("training time for epoch: ", round((time.time() - epoch_start)/60, 3), " minutes", file = training_file)
        print("num no cands: ", num_no_cands, " out of ", len(train_texts), file = training_file)
        print("", file = training_file)

        f1_by_thresh, recall_by_thresh, precision_by_thresh, acc_by_thresh, flipped_proportion_by_thresh, recourse_proportion_by_thresh = \
            evaluate(pos_probs, train_labels, thresholds_to_eval, training_file, negative_by_thresh, flipped_by_thresh, "val")


        print("Train (avg) epoch loss: ", train_epoch_loss/len(train_texts))
        

        training_file.flush()


        model.eval()
            
        val_correct = 0
        epoch_val_loss = 0
        pos_probs = []

        flipped_by_thresh = {thresh: 0 for thresh in thresholds_to_eval}
        negative_by_thresh = {thresh :0 for thresh in thresholds_to_eval}

        for i, (text, label) in tqdm(enumerate(zip(dev_texts, dev_labels)), total = len(dev_texts)):
            input_ids, labels = get_tensors(device, tokenizer, text, label)
            logits, labels, pos_prob = get_pred(model, tokenizer, device, input_ids, labels)            
            _, delta_logits, delta_prob = get_delta_opt(model, tokenizer, device, text, max_candidates)
            epoch_val_loss += combined_loss(model, device, logits, labels, delta_logits, loss_fn, recourse_loss_weight).item()
            
            del input_ids
            del labels
            del logits
            torch.cuda.empty_cache()
            
            pos_probs.append(pos_prob.item())

            for t in thresholds_to_eval:
                if pos_prob.item() < t:
                    negative_by_thresh[t] += 1
                    if delta_prob.item() >= t:
                        flipped_by_thresh[t] += 1
                        
        if epoch_val_loss < best_val_loss:
            best_model_name = weight_dir + str(recourse_loss_weight) + '_best_model.pt'
            torch.save(model, best_model_name)
            best_epoch = True

        else:
            best_epoch = False


        f1_by_thresh, recall_by_thresh, precision_by_thresh, acc_by_thresh, flipped_proportion_by_thresh, recourse_proportion_by_thresh = \
            evaluate(pos_probs, dev_labels, thresholds_to_eval, training_file, negative_by_thresh, flipped_by_thresh, "val")

        if best_epoch:
            best_model_info_file_name = weight_dir + str(recourse_loss_weight) + "_best_model_val_info.txt"
            best_model_info_file = open(best_model_info_file_name, "w")
            print("epoch: ", epoch, file = best_model_info_file)
            best_model_info_file.close()
        
        if best_epoch:

            thresholds_data = {}

            thresholds_data['thresholds'] = thresholds_to_eval
            thresholds_data['f1s'] = f1_by_thresh
            thresholds_data['accs'] = acc_by_thresh
            thresholds_data['recalls'] = recall_by_thresh
            thresholds_data['precisions'] = precision_by_thresh
            thresholds_data['flipped_proportion'] = flipped_proportion_by_thresh
            thresholds_data['recourse_proportion'] = recourse_proportion_by_thresh
            thresholds_df = pd.DataFrame(data=thresholds_data)
            
            best_model_thresholds_file_name = weight_dir + str(recourse_loss_weight) + '_val_thresholds_info.csv'
            thresholds_df.to_csv(best_model_thresholds_file_name, index_label='index')

        training_file.flush()


    training_file.close()

def evaluate(pos_probs, labels, thresholds_to_eval, training_file, negative_by_thresh, flipped_by_thresh, data_stub):

    # if best epoch, eval
    np_probs = np.array(pos_probs)
    np_labels = np.array(labels)

    f1_by_thresh, recall_by_thresh, precision_by_thresh, acc_by_thresh, flipped_proportion_by_thresh, recourse_proportion_by_thresh = [], [], [], [], [], []

    for t_idx, t in enumerate(thresholds_to_eval):
        label_preds = np.array([0.0 if a < t else 1.0 for a in np_probs])

        f1 = round(f1_score(label_preds, np_labels), 3)
        f1_by_thresh.append(f1) 

        recall = round(recall_score(label_preds, np_labels), 3)
        recall_by_thresh.append(recall)

        prec = round(precision_score(label_preds, np_labels), 3)
        precision_by_thresh.append(prec)

        acc = round(np.sum(label_preds == np_labels)/np_labels.shape[0], 3)
        acc_by_thresh.append(acc) 

        num_neg = negative_by_thresh[t]
        num_pos = len(labels) - num_neg
        assert (num_neg + num_pos) == len(labels)
        flipped = flipped_by_thresh[t]

        if num_neg != 0:
            flipped_proportion = round(flipped/num_neg, 3)
        else:
            flipped_proportion = 0

        recourse_proportion = round((flipped + num_pos)/len(labels), 3)

        flipped_proportion_by_thresh.append(flipped_proportion)
        recourse_proportion_by_thresh.append(recourse_proportion)

        print(data_stub + " STATS FOR THRESHOLD = " + str(t) + ": ", file = training_file)
        print(data_stub + " f1: ", f1, file = training_file)
        print(data_stub + " acc: ", acc, file = training_file)
        print(data_stub + " prec: ", prec, file = training_file)
        print(data_stub + " recall: ", recall, file = training_file)
        print(data_stub + " flipped: ", flipped_proportion, file = training_file)
        print(data_stub + " recourse: ", recourse_proportion, file = training_file)
        print("\n")

    return f1_by_thresh, recall_by_thresh, precision_by_thresh, acc_by_thresh, flipped_proportion_by_thresh, recourse_proportion_by_thresh



def run_evaluate(weight_dir, recourse_loss_weight, tokenizer, device, max_candidates = 10):
    model = load_trained_model(weight_dir, recourse_loss_weight)
    model = model.to(device)
    test_texts, test_labels = get_sst_data('data/nlp_data/test.txt')

    pos_probs = []

    thresholds_info = get_threshold_info(weight_dir, recourse_loss_weight)

    thresholds = list(thresholds_info['thresholds'])

    f1s = thresholds_info['f1s'] 
    thresholds_to_eval = [thresholds[np.argmax(f1s)]]

    flipped_by_thresh = {thresh: 0 for thresh in thresholds_to_eval}
    negative_by_thresh = {thresh :0 for thresh in thresholds_to_eval}

    for i, (text, label) in tqdm(enumerate(zip(test_texts, test_labels)), total = len(test_texts)):
        input_ids, labels = get_tensors(device, tokenizer, text, label) 
        logits, labels, pos_prob = get_pred(model, tokenizer, device, input_ids, labels)            
        _, delta_logits, delta_prob = get_delta_opt(model, tokenizer, device, text, max_candidates)
        
        del input_ids
        del labels
        del logits
        torch.cuda.empty_cache()
        
        pos_probs.append(pos_prob.item())

        for t in thresholds_to_eval:
            if pos_prob.item() < t:
                negative_by_thresh[t] += 1
                if delta_prob.item() >= t:
                    flipped_by_thresh[t] += 1

    f1_by_thresh, recall_by_thresh, precision_by_thresh, acc_by_thresh, flipped_proportion_by_thresh, recourse_proportion_by_thresh = \
        evaluate(pos_probs, test_labels, thresholds_to_eval, None, negative_by_thresh, flipped_by_thresh, "test")


    thresholds_data = {}

    thresholds_data['thresholds'] = thresholds_to_eval
    thresholds_data['f1s'] = f1_by_thresh
    thresholds_data['accs'] = acc_by_thresh
    thresholds_data['recalls'] = recall_by_thresh
    thresholds_data['precisions'] = precision_by_thresh
    thresholds_data['flipped_proportion'] = flipped_proportion_by_thresh
    thresholds_data['recourse_proportion'] = recourse_proportion_by_thresh
    thresholds_df = pd.DataFrame(data=thresholds_data)
    
    best_model_thresholds_file_name = weight_dir + str(recourse_loss_weight) + '_test_thresholds_info.csv'
    thresholds_df.to_csv(best_model_thresholds_file_name, index_label='index')



