import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from features.extraction import ActionFeatureGenerator
from models.state import ParsingState
from models.tree import RstTree
from utils.constants import *
from eval.evaluation import Evaluator
from numpy.random import binomial
from utils.other import xidx_action_map, xidx_relation_map, cleanup_load_dict


class NeuralRstParserCoref(object):
    
    def __init__(self, clf, coref_trainer, data_helper, config):
        
        self.config = config
        self.data_helper = data_helper
        
        self.clf = clf
        self.coref_trainer = coref_trainer
        # self.loss = CrossEntropyLoss(reduction='mean')
        # move tensors to gpu for loss:
        self.loss = CrossEntropyLoss(reduction='mean').to(config[DEVICE])
        self.optim = None

    def get_optim_scheduler(self, train_loader):
        no_decay = ['bias', 'LayerNorm.weight']
        self.optim = AdamW(params=[
                                    {'params': [p for n, p in self.clf.bert.named_parameters() 
                                                if not any(nd in n for nd in no_decay)], 
                                     'lr': 1e-05},  # Bert params outside no_decay
                                    {'params': [p for n, p in self.clf.bert.named_parameters() 
                                                if any(nd in n for nd in no_decay)], 
                                     'weight_decay': 0.0, 'lr': 1e-05},  # Bert params in no_decay
                                    {'params': [p for n, p in self.clf.named_parameters()  # Clf outside no_decay
                                               if ("bert" not in n and not any(nd in n for nd in no_decay))]},
                                    {'params': [p for n, p in self.clf.named_parameters()
                                               if ("bert" not in n and any(nd in n for nd in no_decay))],
                                     'weight_decay': 0.0},  # Clf params in no_decay
                                   ],
                                   lr=0.0002, weight_decay=0.01
                          )
        
        self.num_batches = 34685 / self.clf.config[BATCH_SIZE]
        train_steps = int(20 * self.num_batches)

        self.scheduler = get_linear_schedule_with_warmup(self.optim,
                                                          num_warmup_steps=int(train_steps*0.1), 
                                                          num_training_steps=train_steps)

    def train_classifier(self, train_loader):
        
        # Initialize optimizer and scheduler
        self.get_optim_scheduler(train_loader)
        
        if os.path.isfile("../data/model/" + self.config[MODEL_NAME]):
            epoch_start = self.load("../data/model/" + self.config[MODEL_NAME])
        else:
            epoch_start = 0

        for epoch in range(epoch_start+1, 21):
            cost_acc = 0
            self.clf.train()

            print("============ epoch: ", epoch, " ============")
            for i, data in tqdm(enumerate(train_loader)):
                # if 0, train on random datapoint from coref corpus
                while self.config[MODEL_TYPE] > 1 and binomial(1, self.task_p) == 0:
                    cost_acc += self.coref_trainer.train_epoch(i, 1)
                cost_acc += self.train_sample_rst(data)

            print("Total cost for epoch %d is %f" % (epoch, cost_acc))

            print("============ Evaluating on the dev set ============")
            self.save(self.config[MODEL_NAME], epoch)
            self.evaluate()

    def train_sample_rst(self, sample):
        docs, batched_clusters, action_feats, neural_feats, all_actions, all_relations, rel_mask = sample
        
        self.optim.zero_grad()
        
        # Forward pass
        if self.clf.config[MODEL_TYPE] in [0, 3]:
            span_embeds = self.clf.get_edus_bert_coref(docs, [None] * len(docs), neural_feats)
        
        # Compute action loss
        action_probs, rel_probs = self.clf.decode_action_coref(span_embeds, action_feats)
        # cost = self.loss(action_probs, all_actions)
        cost = self.loss(action_probs.to(self.config[DEVICE]), all_actions.to(self.config[DEVICE]))

        # Compute relation loss
        rel_probs, rel_labels = rel_probs[rel_mask], all_relations[rel_mask]
        if rel_labels.shape[0] > 0:
            # cost += self.loss(rel_probs, rel_labels)
            cost += self.loss(rel_probs.to(self.config[DEVICE]), rel_labels.to(self.config[DEVICE]))
        
        # Update the model
        cost.backward()
        nn.utils.clip_grad_norm_(self.clf.parameters(), 1.0)
        self.optim.step()
        self.scheduler.step()
            
        return cost.item()                

    def sr_parse(self, doc, gold_actions, gold_rels):
        # Generate coref clusters for the document
        clusters = None
            
        # Stack/Queue state
        conf = ParsingState([], [], self.clf.config)
        conf.init(doc)        
        all_action_probs, all_rel_probs = [], []
        # Until the tree is built
        while not conf.end_parsing():
            
            # Get features for the current stack/queue state, and span boundaries
            stack, queue = conf.get_status()
            fg = ActionFeatureGenerator(stack, queue, [], doc, self.data_helper, self.config)
            action_feat, span_boundary = fg.gen_features()
            span_embeds = self.clf.get_edus_bert_coref([doc], [clusters], [span_boundary])
            action_probs, rel_probs = self.clf.decode_action_coref(span_embeds, [action_feat])
            all_action_probs.append(action_probs.squeeze())
            sorted_action_idx = torch.argsort(action_probs, descending=True)
            sorted_rel_idx = torch.argsort(rel_probs, descending=True)
            
            # Select Shift/Reduce action (shift/reduce-nn/...)
            action_idx = 0
            pred_action, pred_nuc = xidx_action_map[int(sorted_action_idx[0, action_idx])]      
            while not conf.is_action_allowed((pred_action, pred_nuc, None), doc):
                action_idx += 1
                pred_action, pred_nuc = xidx_action_map[int(sorted_action_idx[0, action_idx])]
                
            # Select Relation annotation
            pred_rel = None
            if pred_action != "Shift":
                all_rel_probs.append(rel_probs.squeeze())
                pred_rel_idx = int(sorted_rel_idx[0, 0])
                pred_rel = xidx_relation_map[pred_rel_idx]
            assert not (pred_action == "Reduce" and pred_rel is None)
            
            predictions = (pred_action, pred_nuc, pred_rel)
            conf.operate(predictions)
            
        # Shift/Reduce loss
        cost = self.loss(torch.stack(all_action_probs), gold_actions)
        
        # Relation annotation loss
        if all_rel_probs:
            cost_relation = self.loss(torch.stack(all_rel_probs), gold_rels)
            cost += cost_relation
        
        tree = conf.get_parse_tree()

        rst_tree = RstTree()
        rst_tree.assign_tree(tree)
        rst_tree.assign_doc(doc)
        rst_tree.back_prop(tree, doc)
        
        return rst_tree, cost.item()

    def evaluate(self):
        self.clf.eval()
        with torch.no_grad():
            eval = Evaluator(self, self.data_helper, self.config)
            eval.eval_parser(self.data_helper.val_trees)

    def save(self, model_name, epoch):
        """Save models
        """
        save_dict = {'epoch': epoch,
                     'model_state_dict': self.clf.state_dict(),
                     'optimizer_state_dict': self.optim.state_dict(),
                     'scheduler_state_dict': self.scheduler.state_dict()
                     }
        torch.save(save_dict, os.path.join("../data/model/", model_name))

    def load(self, model_dir):
        """ Load models
        """
        model_save = torch.load(model_dir)
        cleanup_load_dict(model_save)
        self.clf.load_state_dict(model_save['model_state_dict'])
        self.clf.eval()
        if self.optim is not None:
            self.optim.load_state_dict(model_save['optimizer_state_dict'])
            self.scheduler.load_state_dict(model_save['scheduler_state_dict'])

        return model_save['epoch']
