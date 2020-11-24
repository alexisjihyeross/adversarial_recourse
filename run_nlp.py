import importlib
import os
import torch
from utils.nlp_utils import *
import nltk
nltk.download('wordnet')

weights = [0.0, 0.25]
thresholds_to_eval = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

for weight in weights:
    weight_dir = 'results/nlp/' + str(weight) + "/"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    model, tokenizer = load_pretrained_model(device, model_name = 'bert-base-uncased')

    # train and evaluate
    train_nlp(model, tokenizer, weight_dir, thresholds_to_eval, weight)
    run_nlp_evaluate(weight_dir, weight, tokenizer, device)
