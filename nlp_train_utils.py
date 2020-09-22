import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW

from nltk.tree import Tree
from nltk.tokenize.treebank import TreebankWordDetokenizer

from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import numpy as np

import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from sklearn.metrics import f1_score, precision_score, recall_score


def load_model(device, model_name = 'bert-base-uncased'):
    print("loading model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    model.train()

    tokenizer = BertTokenizer.from_pretrained(model_name)

    print("done.")
    return model, tokenizer

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
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.text, 'lxml')
    return [span.text for span in soup.findAll('a', {'class': 'css-4elvh4'})] # class = .css-7854fb for less relevant


def get_candidates(model, text, max_candidates = 20):
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
            candidates[counter] = (TreebankWordDetokenizer().detokenize([a if x == word else x for x in words]))
            counter += 1
            if counter >= max_candidates:
                return candidates
    return candidates

def get_delta_opt(model, tokenizer, device, text):
    cands = get_candidates(model, text)
    max_prob = 0
    for c in cands:
        cand_logits, cand_labels, cand_prob = get_pred(model, tokenizer, device, c, 1.0)
        if cand_prob > max_prob:
            max_cand = c
            max_prob = cand_prob
            max_logits = cand_logits
            max_prob = cand_prob
        else:
            del cand_logits
            del cand_labels
            del cand_prob
            torch.cuda.empty_cache()
    return max_cand, max_logits, max_prob

def get_pred(model, tokenizer, device, text, label):
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids']
    input_ids = encoding.to(device)
    labels = torch.LongTensor([label]).to(device)
    outputs = model(input_ids)
    logits = outputs[0]
    pos_prob = torch.nn.Softmax(dim=-1)(logits)[:, -1]
    return logits, labels, pos_prob


def train_nlp(weight_dir, thresholds_to_eval, recourse_loss_weight):

    # get data
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model and tokenizer
    model, tokenizer = load_model(device, model_name = 'bert-base-uncased')
    
    train_texts, train_labels = get_sst_data('data/nlp_data/train.txt')
    dev_texts, dev_labels = get_sst_data('data/nlp_data/dev.txt')

    batch_size = 32
    threshold = 0.5

    lr = 2e-5
    num_warmup_steps = 0
    num_epochs = 3
    num_train_steps = len(train_texts)/batch_size * num_epochs

    optim = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps, num_train_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    def combined_loss(model, device, logits, labels, delta_logits, loss_fn, recourse_loss_weight):
        normal_loss = loss_fn(logits, labels)
        recourse_loss = loss_fn(delta_logits, torch.LongTensor([1.0]).to(device))
        return recourse_loss * recourse_loss_weight + normal_loss

    best_val_loss = 100000000

    for epoch in range(num_epochs):
        batch_loss, train_epoch_loss, train_correct = 0, 0, 0
        
        print("EPOCH: ", epoch)
        model.train()

        for i, (text, label) in tqdm(enumerate(zip(train_texts, train_labels)), total = len(train_texts)):
            logits, labels, pos_prob = get_pred(model, tokenizer, device, text, label)
            _, delta_logits, _ = get_delta_opt(model, tokenizer, device, text)
            batch_loss += combined_loss(model, device, logits, labels, delta_logits, loss_fn, recourse_loss_weight)
                        
            if i % 1000 == 0:
                print(i, " out of ", len(train_texts))
            
            if i % batch_size == 0:    
                model.zero_grad() 
                batch_loss.backward()
                train_epoch_loss += batch_loss.item()
                optim.step()
                scheduler.step()
                del batch_loss
                torch.cuda.empty_cache()
                batch_loss = 0
                
        print("Train acc: ", train_correct/len(train_texts))
        print("Train epoch loss: ", train_epoch_loss/len(train_texts))
        
        model.eval()
            
        val_correct = 0
        epoch_val_loss = 0
        pos_probs = []

        flipped_by_thresh = {thresh: 0 for thresh in thresholds_to_eval}
        negative_by_thresh = {thresh :0 for thresh in thresholds_to_eval}

        for i, (text, label) in tqdm(enumerate(zip(dev_texts, dev_labels)), total = len(dev_texts)):
            logits, labels, pos_prob = get_pred(model, tokenizer, device, text, label)
            _, delta_logits, delta_prob = get_delta_opt(model, tokenizer, device, text)
            epoch_val_loss += combined_loss(model, device, logits, labels, delta_logits, loss_fn, recourse_loss_weight)


            del input_ids
            
            pos_probs.append(pos_prob.item())

            for t in thresholds_to_eval:
                if pos_prob.item() < t:
                    negative_by_thresh[t] += 1
                    if delta_prob >= t:
                        flipped_by_thresh[t] += 1
            
        if epoch_val_los < best_val_loss:
            best_model_name = weight_dir + str(recourse_loss_weight) + 'best_model.pt'
            torch.save(model, best_model_name)
            best_epoch = True

        else:
            best_epoch = False

        # if best epoch, eval
        if best_epoch:
            np_probs = np.array(pos_probs)
            np_labels = np.array(dev_labels)

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
                num_pos = len(dev_labels) - num_neg
                assert (num_neg + num_pos) == len(dev_labels)
                flipped = flipped_by_thresh[t]

                if num_neg != 0:
                    flipped_proportion = round(flipped/num_neg, 3)
                else:
                    flipped_proportion = 0

                recourse_proportion = round((flipped + num_pos)/len(dev_labels), 3)

                flipped_proportion_by_thresh.append(flipped_proportion)
                recourse_proportion_by_thresh.append(recourse_proportion)

            
            thresholds_data = {}

            thresholds_data['thresholds'] = thresholds_to_eval
            thresholds_data['precisions'] = precision_by_thresh
            thresholds_data['flipped_proportion'] = flipped_proportion_by_thresh
            thresholds_data['recourse_proportion'] = recourse_proportion_by_thresh
            thresholds_data['f1s'] = f1_by_thresh
            thresholds_data['accs'] = acc_by_thresh
            thresholds_data['recalls'] = recall_by_thresh
            thresholds_data['precisions'] = precision_by_thresh

            thresholds_df = pd.DataFrame(data=thresholds_data)
            best_model_thresholds_file_name = weight_dir + str(recourse_loss_weight) + '_val_thresholds_info.csv'
            thresholds_df.to_csv(best_model_thresholds_file_name, index_label='index')
            
        print("VAL ACC: ", val_correct/len(dev_texts))
        print("+ ", val_preds.count(1.0))
        print("-", val_preds.count(0.0))


