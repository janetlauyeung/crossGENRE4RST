import os, io
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from itertools import chain
from tqdm import tqdm
import networkx as nx
import random
from subprocess import Popen, PIPE

from ubc_coref.utils import *

class Trainer:
    """ Class dedicated to training and evaluating the model
    """
    def __init__(self, model, train_corpus, val_corpus, test_corpus,
                    debug=False, distribute_model=False,
                    pretrained_path=None):
        self.__dict__.update(locals())
        self.debug = debug
        self.train_corpus = list(self.train_corpus)
        self.steps = len(self.train_corpus)
        self.test_corpus = test_corpus
        if self.debug:
            self.val_corpus.docs = [val_corpus.docs[0]]
        self.model = to_cuda(model)
        if distribute_model:
            self.model.encoder.cuda(1)
            self.model.score_spans.cuda(2)

        self.MAX = 2
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer = AdamW(params=[
                                        {'params': [p for n, p in chain(model.score_spans.named_parameters(),
                                                                        model.score_pairs.named_parameters())
                                                   if any(nd in n for nd in no_decay)],
                                                        'weight_decay': 0.0},
                                        {'params': [p for n, p in chain(model.score_spans.named_parameters(),
                                                                        model.score_pairs.named_parameters())
                                                   if not any(nd in n for nd in no_decay)]},
                                        {'params': [p for n, p in model.encoder.named_parameters() 
                                                    if any(nd in n for nd in no_decay)], 
                                                     'weight_decay': 0.0, 'lr': 1e-05},
                                        {'params': [p for n, p in model.encoder.named_parameters() 
                                                    if not any(nd in n for nd in no_decay)], 
                                                     'lr': 1e-05}
                                    ],
                                    lr=0.0002, weight_decay=0.01)
        
        train_steps = 20 * self.steps
        self.scheduler =  get_linear_schedule_with_warmup(self.optimizer, 
                                                          num_warmup_steps=int(train_steps*0.1), 
                                                          num_training_steps=int(train_steps))
        self.start_epoch = 0
        if pretrained_path is not None:
            self.load_model(pretrained_path)
        

    def train(self, num_epochs, eval_interval=1, *args, **kwargs):
        """ Train a model """
        for epoch in range(self.start_epoch + 1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)

            # Save often
            self.save_model("coref_model_saves/coref_model.pt", epoch)
            torch.cuda.empty_cache()
            # Evaluate every eval_interval epochs
            if epoch % eval_interval == 0:
                print('\n\nEVALUATION\n\n')
                with torch.no_grad():
                    self.model.eval()
                    results = self.evaluate(self.val_corpus)
                    print(results)

    def train_epoch(self, epoch):
        """ Run a training epoch over 'steps' documents """

        # Set model to train (enables dropout)
        self.model.train()
        # Randomly sample documents from the train corpus
        batch = random.sample(self.train_corpus, self.steps)
        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []
        for i, document in enumerate(tqdm(batch)):
            
            # Randomly s document to up to 50 sentences
            doc = document.truncate(True, self.MAX)

            # Compute loss, number gold links found, total gold links
            loss, mentions_found, total_mentions, \
                corefs_found, total_corefs, corefs_chosen = self.train_doc(doc)
            
            # Track stats by document for debugging
            print(document, '| Loss: %f | Mentions: %d/%d | Coref recall: %d/%d | Corefs precision: %d/%d' \
                % (loss, mentions_found, total_mentions,
                    corefs_found, total_corefs, corefs_chosen, total_corefs))

            epoch_loss.append(loss)
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            epoch_corefs.append(safe_divide(corefs_found, total_corefs))
            epoch_identified.append(safe_divide(corefs_chosen, total_corefs))
            
            # Step the learning rate decrease scheduler
            self.scheduler.step()
            
        print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref precision: %f \n' \
                % (epoch, np.mean(epoch_loss), np.mean(epoch_mentions),
                    np.mean(epoch_corefs), np.mean(epoch_identified)))

    def train_doc(self, document):
        """ Compute loss for a forward pass over a document """
        # Extract gold coreference links
        gold_corefs, total_corefs, \
            gold_mentions, total_mentions = extract_gold_corefs(document)
        # Zero out optimizer gradients
        self.optimizer.zero_grad()
        # Init metrics
        mentions_found, corefs_found, corefs_chosen = 0, 0, 0
        
        # Predict coref probabilites for each span in a document
        spans, scores, _ = self.model(document)
        
        # If too few spans were found
        if spans is None:
            return (0, 0, 0, 0, 0, 0)
        
        # Get log-likelihood of correct antecedents implied by gold clustering
        gold_indexes = to_cuda(torch.zeros_like(scores))
        for idx, span in enumerate(spans):
            
            # Log number of mentions found
            if (span.i1, span.i2) in gold_mentions:
                mentions_found += 1
                # Check which of these tuples are in the gold set, if any
                golds = [
                    i+1 for i, link in enumerate(span.yi_idx)
                    if link in gold_corefs
                ]

                # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
                if golds:
                    gold_indexes[idx, golds] = 1

                    # Progress logging for recall
                    corefs_found += len(golds)
                    found_corefs = sum((scores[idx, golds] > scores[idx, 0])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    # Otherwise, set gold to dummy
                    gold_indexes[idx, 0] = 1

            else:
                # Otherwise, set gold to dummy
                gold_indexes[idx, 0] = 1

        loss = self.softmax_loss(scores, gold_indexes)
                         
        # Backpropagate
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Step the optimizer
        self.optimizer.step()

        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, corefs_chosen)
    
    def softmax_loss(self, scores, gold_indexes):
        gold_scores = scores + torch.log(gold_indexes)
        marginalized_gold_scores = torch.logsumexp(gold_scores, 1) 
        log_norm = torch.logsumexp(scores, 1) 
        loss = log_norm - marginalized_gold_scores
        
        return torch.sum(loss)   
                         
    def evaluate(self, corpus, eval_script='../ubc_coref/eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        # Predict files
        print('Evaluating the model...')
        predicted_docs = [self.predict(doc) for doc in tqdm(corpus)]
        
        corpus.docs = predicted_docs

        # Output results
        golds_file, preds_file = self.to_conll(corpus, eval_script)

        # Run perl script
        print('Running Perl evaluation script...')
        p = Popen([eval_script, 'all', golds_file, preds_file, "none"], stdout=PIPE)
        stdout, stderr = p.communicate()
        print("All results: ", str(stdout))
        results = str(stdout)
        # Write the results out for later viewing
        with open('../preds/results.txt', 'w+') as f:
            f.write(results)
            f.write('\n\n\n')
        
        return results
        

    def predict(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()

        # Pass the document through the model
        spans, probs, _ = self.model(doc)

        # Cluster found coreference links
        for i, span in enumerate(spans):
            
            found_coref = torch.argmax(probs[i, :])
            if probs[i, found_coref] > 0:
                
                link = spans[i].yi[found_coref - 1]
                graph.add_edge((link.i1, link.i2), (span.i1, span.i2))
                        
        # Extract clusters as nodes that share an edge
        clusters = list(nx.connected_components(graph))

        # Initialize token tags
        token_tags = [[] for _ in range(len(doc))]

        # Add in cluster ids for each cluster of corefs in place of token tag
        for idx, cluster in enumerate(clusters):
            for i1, i2 in cluster:

                if i1 == i2:
                    token_tags[i1].append(f'({idx})')

                else:
                    token_tags[i1].append(f'({idx}')
                    token_tags[i2].append(f'{idx})')

        doc.tags = ['|'.join(t) if t else '-' for t in token_tags]

        return doc
    
    def predict_clusters(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()

        # Pass the document through the model
        spans, probs, embeds = self.model(doc)

        # Cluster found coreference links
        for i, span in enumerate(spans):
            
            found_coref = torch.argmax(probs[i, :])
            if probs[i, found_coref] > 0:
                
                link = spans[i].yi[found_coref - 1]
                graph.add_edge((link.i1, link.i2), (span.i1, span.i2))
                        
        # Extract clusters as nodes that share an edge
        clusters = list(nx.connected_components(graph))
       
        return clusters, embeds


    def to_conll(self, val_corpus, eval_script):
        """ Write to out_file the predictions, return CoNLL metrics results """

        # Make predictions directory if there isn't one already
        golds_file, preds_file = '../preds/golds.txt', '../preds/predictions.txt'
        if not os.path.exists('../preds/'):
            os.makedirs('../preds/')

        # Combine all gold files into a single file (Perl script requires this)
        golds_file_content = flatten([doc.raw_text for doc in val_corpus])
        with io.open(golds_file, 'w', encoding='utf-8', errors='strict') as f:
            for line in golds_file_content:
                f.write(line)

        # Dump predictions
        with io.open(preds_file, 'w', encoding='utf-8', errors='strict') as f:

            for doc in val_corpus:

                current_idx = 0

                for line in doc.raw_text:

                    # Indicates start / end of document or line break
                    if line.startswith('#begin') or line.startswith('#end') or line == '\n':
                        f.write(line)
                        continue
                    else:
                        # Replace the coref column entry with the predicted tag
                        tokens = line.split()
                        tokens[-1] = doc.tags[current_idx]

                        # Increment by 1 so tags are still aligned
                        current_idx += 1

                        # Rewrite it back out
                        f.write('    '.join(tokens))
                    f.write('\n')

        return golds_file, preds_file

    def save_model(self, savepath, epoch):
        """ Save model state dictionary """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            }, savepath + '_' + str(epoch) + '_.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state_dict = torch.load(loadpath)
        self.start_epoch = model_save['epoch']
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
