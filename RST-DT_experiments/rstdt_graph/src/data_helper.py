import os
import pickle
import sys
import numpy as np
from models.tree import RstTree
from utils.document import Doc
from trankit import Pipeline
from ubc_coref.loader import Document
from sklearn.model_selection import train_test_split
from ubc_coref import loader
from utils.other import action_map, relation_map
sys.modules['loader'] = loader

p = Pipeline('english', gpu=True, cache_dir='./cache')  # initialize a pipeline for English


class DataHelper(object):
    
    def __init__(self):
        self.action_map = {}
        self.relation_map = {}

    def create_data_helper(self, data_dir, config, coref_trainer):        
        print("Parsing trees")
        
        # read train data
        all_feats_list, self.feats_list = [], []
        all_actions_numeric, self.actions_numeric = [], []
        all_relations_numeric, self.relations_numeric = [], []
        self.docs = []
        self.val_trees = []
        
        print("Generating features")
        for i, rst_tree in enumerate(self.read_rst_trees(data_dir=data_dir)):
            feats, actions, relations = rst_tree.generate_action_relation_samples(config)
            fdis = feats[0][0]
            
            # Old doc instance for storing sentence/paragraph/document features
            doc = Doc()
            eval_instance = fdis.replace('.dis', '.merge')
            doc.read_from_fmerge(eval_instance)

            # updated: use trankit for tokenization instead of nltk
            tok_edus = [tokenize_and_extract_tokens(edu) for edu in doc.doc_edus]
            tokens = flatten(tok_edus)
            
            # Coreference resolver document instance for coreference functionality
            # (converting tokens to wordpieces and getting corresponding coref boundaries etc)
            coref_document = Document(raw_text=None, tokens=tokens, sents=tok_edus, corefs=[],
                                      speakers=["0"] * len(tokens), genre="nw", filename=fdis)
            # Duplicate for convenience
            coref_document.token_dict = doc.token_dict
            coref_document.edu_dict = doc.edu_dict
            coref_document.old_doc = doc
            
            for (feat, action, relation) in zip(feats, actions, relations):
                feat[0] = i
                all_feats_list.append(feat)
                all_actions_numeric.append(action)
                all_relations_numeric.append(relation)
                                        
            self.docs.append(coref_document)

            if i % 50 == 0:
                print("Processed ", i + 1, " trees")

        assert len(all_feats_list) == len(all_actions_numeric) == len(all_relations_numeric), \
            f"Unequal number of feature list, action item, and relation label for {all_feats_list[0][0].split(os.sep)[-1]}!"

        all_actions_numeric = [action_map[x] for x in all_actions_numeric]
        all_relations_numeric = [relation_map[x] for x in all_relations_numeric]
        
        # Select only those stack-queue actions that belong to trees in the train set 
        for i, feat in enumerate(all_feats_list):
            # if feat[0] in train_indexes:
                # print(f"{i+1}\tLENGTH OF FEAT: {len(feat[1][0])}\n{feat}")        # len(feat[1][0]): 30, 23, 16
                self.feats_list.append(feat)
                self.actions_numeric.append(all_actions_numeric[i])
                self.relations_numeric.append(all_relations_numeric[i])
                            
        # self.val_trees = [self.docs[index].filename for index in val_indexes]
        self.val_trees = [os.path.join('../data/dev_dir/', f) for f in os.listdir(f'../data/dev_dir/') if f.endswith(".dis")]
        self.all_clusters = []
            
    def save_data_helper(self, fname):
        print('Save data helper...')
        data_info = {
            'feats_list': self.feats_list,
            'actions_numeric': self.actions_numeric,
            'relations_numeric': self.relations_numeric,
            'docs': self.docs,
            'val_trees': self.val_trees,
            'all_clusters': self.all_clusters,
        }
        
        with open(fname, 'wb') as fout:
            pickle.dump(data_info, fout)

    def load_data_helper(self, fname):
        print('Load data helper ...')
        with open(fname, 'rb') as fin:
            data_info = pickle.load(fin)
        self.feats_list = data_info['feats_list']
        self.actions_numeric = data_info['actions_numeric']  
        self.relations_numeric = data_info['relations_numeric'] 
        self.val_trees = data_info['val_trees']
        self.docs = data_info['docs']
        self.all_clusters = data_info['all_clusters']
        
    def gen_action_train_data(self, trees):
        return self.feats_list, self.action_seqs_numeric
                
    @staticmethod
    def read_rst_trees(data_dir):
        # Read RST tree file
        files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dis')]
        for i, fdis in enumerate(files):
            fmerge = fdis.replace('.dis', '.merge')
            if not os.path.isfile(fmerge):
                print("Corresponding .fmerge file does not exist. Skipping the file.")
                continue
            rst_tree = RstTree(fdis, fmerge)
            rst_tree.build()
            yield rst_tree
           
        
def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]
        
    
def get_stratify_classes(action_labels):
    
    all_classes = np.array([50, 100, 200])
    stratify_classes = [np.sum(action_label > all_classes) for action_label in action_labels]
    return stratify_classes


def tokenize_and_extract_tokens(edu: str) -> list:
    """
    Tokenize an EDU using trankit: # https://trankit.readthedocs.io/en/latest/tokenize.html
    :param edu: a single edu in a given .edus file
    :return: a list of tokens
    """
    trankit_out = p.tokenize(edu, is_sent=True)  # a dictionary
    tokens = [item["text"] for item in trankit_out["tokens"]]
    return tokens
