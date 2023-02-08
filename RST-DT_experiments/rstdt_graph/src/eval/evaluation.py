import os
import re
import io
import sys
from models.tree import RstTree
from eval.metrics import Metrics
from features.rst_dataset import *
from utils.document import Doc
from utils.tree2rs3 import to_rs3
from ubc_coref.loader import Document
from utils.other import action_map, relation_map

# updated tokenizer
from data_helper import tokenize_and_extract_tokens
from trankit import Pipeline
p = Pipeline('english', gpu=True, cache_dir='./cache')  # initialize a pipeline for English


PY3 = sys.version_info[0] > 2
sys.setrecursionlimit(10**6)


class Evaluator(object):
    def __init__(self, parser, data_helper, config, model_dir='../data/model'):
        print('Load parsing models ...')
        # clf.eval()
        self.parser = parser
        self.data_helper = data_helper
        self.config = config

    def parse(self, doc):
        """ Parse one document using the given parsing models"""
        pred_rst = self.parser.sr_parse(doc)
        return pred_rst

    @staticmethod
    def writebrackets(fname, brackets):
        """ Write the bracketing results into file"""
        with open(fname, 'w') as fout:
            print("Writing to ", fname)
            for item in brackets:
                fout.write(str(item) + '\n')
                
    def eval_parser(self, dev_data=None, path='./examples', save_preds=True, report=False, validation=False, use_parseval=True):
        """ Test the parsing performance"""
        # Evaluation
        met = Metrics(levels=['span', 'nuclearity', 'relation'], use_parseval=use_parseval)
        # ----------------------------------------
        # Read all files from the given path
        if dev_data is None:        # check for dev data before reassigning to it
            dev_data = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.dis')]

        total_cost = 0
        for eval_instance in dev_data:
            # ----------------------------------------
            fmerge = eval_instance.replace('.dis', '.merge')

            doc = Doc()
            doc.read_from_fmerge(fmerge)

            gold_rst = RstTree(eval_instance, fmerge)
            gold_rst.build()

            tok_edus = [tokenize_and_extract_tokens(edu) for edu in doc.doc_edus]
            tokens = flatten(tok_edus)

            coref_document = Document(raw_text=None, tokens=tokens, sents=tok_edus, 
                                      corefs=[], speakers=["0"] * len(tokens), genre="nw", filename=None)

            coref_document.token_dict = doc.token_dict
            coref_document.edu_dict = doc.edu_dict
            doc = coref_document
                
            gold_action_seq, gold_rel_seq = gold_rst.decode_rst_tree()
            
            gold_action_seq = [action_map[x] for x in gold_action_seq]
            gold_relation_seq = [relation_map[x] for x in gold_rel_seq if x is not None]

            # pred_rst: <models.tree.RstTree object at 0x2aedfb3bbcf8>
            pred_rst, cost = self.parser.sr_parse(doc,
                                                  torch.cuda.LongTensor(gold_action_seq),
                                                  torch.cuda.LongTensor(gold_relation_seq))
            total_cost += cost
            
            if save_preds:
                if not os.path.isdir('../data/predicted_trees'):
                    os.mkdir('../data/predicted_trees')

                if not os.path.isdir('../data/predicted_trees_rs3'):
                    os.mkdir('../data/predicted_trees_rs3')

                filename = eval_instance.split(os.sep)[-1]

                # added for writing rs3 files
                doc = Doc()
                doc.read_from_fmerge(fmerge)
                write_rs3(f'../data/predicted_trees_rs3/{filename.split(".")[0]}', doc, pred_rst)

                filepath_p = f'../data/predicted_trees/pred_{filename}'
                filepath_g = f'../data/predicted_trees/gold_{filename}'
                # filepath = f'../data/predicted_trees/{self.config[MODEL_NAME]}_{filename}'

                pred_brackets = pred_rst.bracketing()
                gold_brackets = gold_rst.bracketing()

                # Write brackets into file: PRED
                Evaluator.writebrackets(filepath_p, pred_brackets)
                # Write brackets into file: GOLD
                Evaluator.writebrackets(filepath_g, gold_brackets)

            # ----------------------------------------
            # Evaluate with gold RST tree
            met.eval(gold_rst, pred_rst)
            
        print(f"Total cost: {total_cost}\n")

        if use_parseval:
            print(f"Reporting original Parseval metric.\n")
        else:
            print(f"Reporting RST Parseval metric.\n")

        if report:
            met.report()

        if validation:
            return met.val_criterion()


def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]


def write_rs3(docname, doc, pred_rst):
    with io.open(docname + ".rs3", 'w', encoding="utf8", newline="\n") as f:
        edus = []
        for edu in sorted([e for e in doc.edu_dict]):
            tokids = doc.edu_dict[edu]
            edu = []
            for tok in tokids:
                edu.append(doc.token_dict[tok].word)
            edus.append(" ".join(edu))
        notext = pred_rst.parse()
        for i, edu in enumerate(edus):
            edu = edu.replace("(", "-LRB-").replace("[", "-LSB-").replace(")", "-RRB-").replace("]", "-RSB-")
            notext = notext.replace("( EDU " + str(i+1) + " )", "["+str(i+1)+": " + edu + " ]")
        withtext = re.sub(r'(\( (SN|NN|NS))', r'\n\1', notext)
        output = ""
        indent = 0
        max_id = 1000
        node_id = 0
        for c in withtext:
            if c == "(":
                output += indent * "  "
                indent += 1
                node_id = str(max_id) + ": "
                max_id += 1
            else:
                node_id = ""
            if c == ")":
                indent -= 1
            output += c + node_id

        output = to_rs3(output)
        if PY3:
            f.write(output)
        else:
            f.write(unicode(output.decode("utf8")))
