import importlib
import os
import torch
from nlp_utils import *

recourse_loss_weight = 0.1
weight_dir = 'test_nlp/new/' + str(recourse_loss_weight) + "/"
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
    
thresholds_to_eval = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

model, tokenizer = load_model(device, model_name = 'bert-base-uncased')

train_nlp(model, tokenizer, weight_dir, thresholds_to_eval, recourse_loss_weight)
run_evaluate(weight_dir, recourse_loss_weight, tokenizer, device)
