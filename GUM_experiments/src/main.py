import argparse
import pickle
import random

import numpy as np

from data_helper import DataHelper
from eval.evaluation import Evaluator
from models.classifiers import NeuralClassifier

from ubc_coref.trainer import Trainer
from ubc_coref.coref_model import CorefScore
from ubc_coref import loader

from models.parser_coref import NeuralRstParserCoref
from features.rst_dataset import RstDatasetCoref

import torch
from torch.utils.data import DataLoader
from utils.constants import *
from utils.other import collate_samples
import os

CUDA_LAUNCH_BLOCKING = 1
np.random.seed(42)
random.seed(42)

torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true',
                        help='whether to extract feature templates, action maps and relation maps')
    parser.add_argument('--train', action='store_true',
                        help='whether to train new models')
    parser.add_argument('--eval', action='store_true',
                        help='whether to do evaluation')
    parser.add_argument('--train_dir', help='train data directory')
    parser.add_argument('--eval_dir', help='eval data directory')
    parser.add_argument('--model_type', help='baseline/coref_feats/multitask/multitask-plain - 0/1/2/3', default=0)
    parser.add_argument('--model_name', help='Name of the model')
    parser.add_argument('--pretrained_coref_path', help='Path to the pretrained coref model')
    parser.add_argument('--use_parseval', help='Whether or not to use original Parseval instead of RST-Parseval', action='store_true')
    
    return parser.parse_args()


def get_train_loader(data_helper, config):
    action_feats = data_helper.feats_list
    action_labels = list(zip(data_helper.actions_numeric, 
                        data_helper.relations_numeric))
        
    train_data = RstDatasetCoref(action_feats, action_labels, data_helper, is_train=True)
    
    train_loader = DataLoader(train_data, 
                               batch_size=config[BATCH_SIZE], 
                               shuffle=True,
                               collate_fn=lambda x: collate_samples(data_helper, x), 
                               drop_last=False)
    
    return train_loader


def get_coref_resolver(config):
    
    coref_trainer = None
        
    return coref_trainer


def get_discourse_parser(data_helper, config):
    clf = NeuralClassifier(data_helper, config)
    clf.eval()
    clf.to(config[DEVICE])
    coref_trainer = get_coref_resolver(config)
    rst_parser = NeuralRstParserCoref(clf, coref_trainer, data_helper, config)
    return rst_parser


def train_model_coref(data_helper, config):
    data_helper.load_data_helper(HELPER_PATH)
    train_loader = get_train_loader(data_helper, config)

    os.makedirs('../data/model/', exist_ok=True)
    rst_parser = get_discourse_parser(data_helper, config)
    rst_parser.train_classifier(train_loader)
    
    
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    args = parse_args()
    config = {
        OP_FEATS: False,
        ORG_FEATS: True,
        HIDDEN_DIM: 512,
        BATCH_SIZE: 1,
        DEVICE: "cuda:0",
        KEEP_BOUNDARIES: False,
        DO_COREF: False,
        MODEL_TYPE: int(args.model_type),
        MODEL_NAME: args.model_name,
        PRETRAINED_COREF_PATH: args.pretrained_coref_path
    }
    print("Args:", args)
    print("Config:", config)
    
    data_helper = DataHelper()
    train_dirname = (args.train_dir[:-1] if args.train_dir[-1] == os.sep else args.train_dir).split(os.sep)[-1]
    HELPER_PATH = f"..{os.sep}data{os.sep}{train_dirname}_data_helper_rst.bin"
    print("Helper path:", HELPER_PATH)
    
    if args.prepare:
        # Create training data
        # coref_model = CorefScore(higher_order=True).to(config[DEVICE])
        coref_model = CorefScore().to(config[DEVICE])
        
        coref_trainer = Trainer(coref_model, [], [], [], debug=False)
        
        data_helper.create_data_helper(args.train_dir, config, coref_trainer)
        data_helper.save_data_helper(HELPER_PATH)
            
    if args.train:
        train_model_coref(data_helper, config)
    
    if args.eval:
        # Evaluate models on the RST-DT test set
        data_helper.load_data_helper(HELPER_PATH)
        
        parser = get_discourse_parser(data_helper, config)
        parser.load('../data/model/' + config[MODEL_NAME])
        print("Evaluating")
        with torch.no_grad():
            evaluator = Evaluator(parser, data_helper, config)
            evaluator.eval_parser(None, path=args.eval_dir, save_preds=True, use_parseval=True)
