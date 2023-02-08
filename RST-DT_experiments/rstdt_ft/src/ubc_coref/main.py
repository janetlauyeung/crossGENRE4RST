import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ubc_coref.coref_model import CorefScore
from ubc_coref.trainer import Trainer
from ubc_coref.loader import *

import os
import math
import argparse
import numpy as np
from random import sample
from datetime import datetime
from subprocess import Popen, PIPE
from boltons.iterutils import pairwise
        
parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
parser.add_argument('--distribute_model', action='store_true', default=False, 
                    help='Whether or not to spread the model across 3 GPUs')
parser.add_argument('--pretrained_coref_path', default=None, 
                    help='Path to pretrained model')
parser.add_argument('--train', action='store_true', default=False, help='Train model')
parser.add_argument('--test', action='store_true', default=False, help='Test model')

args = parser.parse_args()

if args.debug:
    print("Debug mode!")
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    torch.autograd.set_detect_anomaly(True)
    eval_interval = 100
else:
    eval_interval = 3

# Initialize model, train
model = CorefScore(distribute_model=args.distribute_model)

if args.train:
    train_corpus, val_corpus = load_corpus_portion("train"), load_corpus_portion("val")
    trainer = Trainer(model, train_corpus, val_corpus, 
                      None, debug=args.debug, 
                      distribute_model=args.distribute_model,
                      pretrained_path=args.pretrained_coref_path)
    trainer.train(20, eval_interval=eval_interval)
elif args.test:
    test_corpus = load_corpus_portion("test")
    trainer = Trainer(model, [], [], 
                      test_corpus, debug=args.debug, 
                      distribute_model=args.distribute_model,
                      pretrained_path=args.pretrained_coref_path)
    trainer.evaluate(test_corpus)
