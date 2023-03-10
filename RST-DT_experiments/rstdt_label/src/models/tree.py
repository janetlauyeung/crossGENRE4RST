import sys
from os.path import isfile
from features.extraction import ActionFeatureGenerator
from models.state import ParsingState
from nltk import Tree
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
from utils.document import Doc
from utils.other import rel2class
from utils.span import SpanNode
from utils.constants import *
from utils.buildtree import getparse


class RstTree(object):
    def __init__(self, fdis=None, fmerge=None):
        self.fdis = fdis
        self.fmerge = fmerge
        self.binary = True
        self.tree, self.doc = None, None

    def assign_tree(self, tree):
        """ Assign a tree instance from external resource
        """
        self.tree = tree

    def assign_doc(self, doc):
        """ Assign a doc instance from external resource
        """
        self.doc = doc

    def build(self):
        """ Build BINARY RST tree
        """
        with open(self.fdis) as fin:
            text = fin.read()
            print(f"Processing {self.fdis}")        # add print file to indicate tree being processed
        # Build RST as annotation
        self.tree = RstTree.build_tree(text)

        # Binarize it
        self.tree = RstTree.binarize_tree(self.tree)
        # Read doc file
        if isfile(self.fmerge):
            doc = Doc()
            doc.read_from_fmerge(self.fmerge)
            self.doc = doc
        else:
            raise IOError("File doesn't exist: {}".format(self.fmerge))
        RstTree.back_prop(self.tree, self.doc)

    def generate_action_relation_samples(self, config):
        """ Generate action samples from an binary RST tree
        :type bcvocab: dict
        :param bcvocab: brown clusters of words
        """
        # Parsing actions and relations
        actions, relations = self.decode_rst_tree()
        # Initialize queue and stack
        queue = RstTree.get_edu_node(self.tree)
        stack = []
        # Start simulating the shift-reduce parsing
        sr_parser = ParsingState(stack, queue)
        action_list = []
        all_features = []
        
        for idx, action in enumerate(actions):
            stack, queue = sr_parser.get_status()
            # Generate features
            fg = ActionFeatureGenerator(stack, queue, actions[:idx], self.doc, None, config)
            action_feats = fg.gen_features()
            all_features.append([self.fdis, action_feats])
            action = (*action, relations[idx])
            sr_parser.operate(action)

        return all_features, actions, relations
    
    def decode_rst_tree(self):
        """ Decoding Shift-reduce actions and span relations from a binary RST tree
        """
        # Start decoding
        post_nodelist = RstTree.postorder_DFT(self.tree, [])
        action_list = []
        relation_list = []
        for node in post_nodelist:
            if (node.lnode is None) and (node.rnode is None):
                action_list.append(('Shift', None))
                relation_list.append(None)
            elif (node.lnode is not None) and (node.rnode is not None):
                form = node.form
                if form == 'NS':
                    relation = RstTree.extract_relation(node.rnode.relation)
                else:
                    relation = RstTree.extract_relation(node.lnode.relation)
                action_list.append(('Reduce', form))
                relation_list.append(relation)
            else:
                raise ValueError("Can not decode Shift-Reduce action")

        return action_list, relation_list

    def convert_node_to_str(self, node, sep=' '):
        text = node.text
        words = [self.doc.token_dict[tidx].word for tidx in text]
        return sep.join(words)

    @staticmethod
    def get_edu_node(tree):
        """ Get all left nodes. It can be used for generating training
            examples from gold RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        # Post-order depth-first traversal
        post_nodelist = RstTree.postorder_DFT(tree, [])
        # EDU list
        edulist = []
        for node in post_nodelist:
            if (node.lnode is None) and (node.rnode is None):
                edulist.append(node)
        return edulist

    @staticmethod
    def build_tree(text):
        """ Build tree from *.dis file

        :type text: string
        :param text: RST tree read from a *.dis file
        """
        tokens = text.strip().replace('//TT_ERR', '').replace('\n', '').replace('(', ' ( ').replace(')', ' ) ').split()
        queue = RstTree.process_text(tokens)
        stack = []
        while queue:
            token = queue.pop(0)
            if token == ')':
                # If ')', start processing
                content = []  # Content in the stack
                while stack:
                    cont = stack.pop()
                    if cont == '(':
                        break
                    else:
                        content.append(cont)
                content.reverse()  # Reverse to the original (stack) order
                # Parse according to the first content word
                if len(content) < 2:
                    raise ValueError("content = {}".format(content))
                label = content.pop(0)
                if label in ['Root', 'Nucleus', 'Satellite']:
                    node = SpanNode(prop=label)
                    node.create_node(content)
                    stack.append(node)
                elif label == 'span':
                    # Merge
                    beginindex = int(content.pop(0))
                    endindex = int(content.pop(0))
                    stack.append(('span', beginindex, endindex))
                elif label == 'leaf':
                    # Merge
                    eduindex = int(content.pop(0))
                    RstTree.check_content(label, content)
                    stack.append(('leaf', eduindex, eduindex))
                elif label == 'rel2par':
                    # Merge
                    relation = content.pop(0)
                    RstTree.check_content(label, content)
                    stack.append(('relation', relation))
                elif label == 'text':
                    # Merge
                    txt = RstTree.create_text(content)
                    stack.append(('text', txt))
                else:
                    raise ValueError(
                        "Unrecognized parsing label: {} \n\twith content = {}\n\tstack={}\n\tqueue={}".format(label,
                                                                                                              content,
                                                                                                              stack,
                                                                                                              queue))
            else:
                # else, keep push into the stack
                stack.append(token)
        return stack[-1]

    @staticmethod
    def process_text(tokens):
        """ Preprocessing token list for filtering '(' and ')' in text,
            replaces them with -LB- and -RB- respectively
        :type tokens: list
        :param tokens: list of tokens
        """
        identifier = '_!'
        within_text = False
        for (idx, tok) in enumerate(tokens):
            if identifier in tok:
                for _ in range(tok.count(identifier)):
                    within_text = not within_text
            if ('(' in tok) and within_text:
                tok = tok.replace('(', '-LB-')
            if (')' in tok) and within_text:
                tok = tok.replace(')', '-RB-')
            tokens[idx] = tok
        return tokens

    @staticmethod
    def create_text(lst):
        """ Create text from a list of tokens

        :type lst: list
        :param lst: list of tokens
        """
        newlst = []
        for item in lst:
            item = item.replace("_!", "")
            newlst.append(item)
        text = ' '.join(newlst)
        # Lower-casing: WHY?
        # return text.lower()
        return text

    @staticmethod
    def check_content(label, c):
        """ Check whether the content is legal

        :type label: string
        :param label: parsing label, such 'span', 'leaf'

        :type c: list
        :param c: list of tokens
        """
        if len(c) > 0:
            raise ValueError("{} with content={}".format(label, c))

    @staticmethod
    def binarize_tree(tree):
        """ Convert a general RST tree to a binary RST tree

        :type tree: instance of SpanNode
        :param tree: a general RST tree
        """
        queue = [tree]
        while queue:
            node = queue.pop(0)
            queue += node.nodelist
            # Construct binary tree
            if len(node.nodelist) == 2:
                node.lnode = node.nodelist[0]
                node.rnode = node.nodelist[1]
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node
            elif len(node.nodelist) > 2:
                # Remove one node from the nodelist
                node.lnode = node.nodelist.pop(0)
                newnode = SpanNode(node.nodelist[0].prop)
                # All remaining 1: children of node
                # are children of newnode
                newnode.nodelist += node.nodelist
                # Right-branching
                node.rnode = newnode
                # Parent node
                # newnode doesn't have parent!
                node.lnode.pnode = node
                node.rnode.pnode = node
                # Add to the head of the queue
                # So the code will keep branching
                # until the nodelist size is 2
                queue.insert(0, newnode)
            # Clear nodelist for the current node
            node.nodelist = []
        return tree

    @staticmethod
    def back_prop(tree, doc):
        """ Starting from leaf node, propagating node
            information back to root node

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        tree_nodes = RstTree.BFTbin(tree)
        tree_nodes.reverse()
        for node in tree_nodes:
            if (node.lnode is not None) and (node.rnode is not None):
                # Non-leaf node
                node.edu_span = RstTree.__getspaninfo(node.lnode, node.rnode)
                node.text = RstTree.__gettextinfo(doc.edu_dict, node.edu_span)
                if node.relation is None:
                    # If it is a new node created by binarization
                    if node.prop == 'Root':
                        pass
                    else:
                        node.relation = RstTree.__getrelationinfo(node.lnode, node.rnode)
                node.form, node.nuc_span, node.nuc_edu = RstTree.__getforminfo(node.lnode, node.rnode)
            elif (node.lnode is None) and (node.rnode is not None):
                raise ValueError("Unexpected left node")
            elif (node.lnode is not None) and (node.rnode is None):
                raise ValueError("Unexpected right node")
            else:
                # Leaf node
                node.text = RstTree.__gettextinfo(doc.edu_dict, node.edu_span)
            
    @staticmethod
    def BFTbin(tree):
        """ Breadth-first traversal on binary RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        queue = [tree]
        bft_nodelist = []
        while queue:
            node = queue.pop(0)
            bft_nodelist.append(node)
            if node.lnode is not None:
                queue.append(node.lnode)
            if node.rnode is not None:
                queue.append(node.rnode)
        return bft_nodelist

    @staticmethod
    def postorder_DFT(tree, nodelist):
        """ Post order traversal on binary RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree

        :type nodelist: list
        :param nodelist: list of node in post order
        """
        if tree.lnode is not None:
            RstTree.postorder_DFT(tree.lnode, nodelist)
        if tree.rnode is not None:
            RstTree.postorder_DFT(tree.rnode, nodelist)
        nodelist.append(tree)
        return nodelist

    @staticmethod
    def __getspaninfo(lnode, rnode):
        """ Get span size for parent node

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        try:
            edu_span = (lnode.edu_span[0], rnode.edu_span[1])
            return edu_span
        except TypeError:
            print(lnode.prop, rnode.prop)
            print(lnode.nuc_span, rnode.nuc_span)
            sys.exit()

    @staticmethod
    def __getforminfo(lnode, rnode):
        """ Get Nucleus/Satellite form and Nucleus span

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        if (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
            nuc_span = lnode.edu_span
            nuc_edu = lnode.nuc_edu
            form = 'NS'
        elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
            nuc_span = rnode.edu_span
            nuc_edu = rnode.nuc_edu
            form = 'SN'
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
            nuc_span = (lnode.edu_span[0], rnode.edu_span[1])
            nuc_edu = lnode.nuc_edu
            form = 'NN'
        else:
            raise ValueError("")
        return form, nuc_span, nuc_edu

    @staticmethod
    def __getrelationinfo(lnode, rnode):
        """ Get relation information

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        if (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
            relation = lnode.relation
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
            relation = rnode.relation
        elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
            relation = lnode.relation
        else:
            print('lnode.prop = {}, lnode.edu_span = {}'.format(lnode.prop, lnode.edu_span))
            print('rnode.prop = {}, lnode.edu_span = {}'.format(rnode.prop, rnode.edu_span))
            raise ValueError("Error when find relation for new node")
        return relation

    @staticmethod
    def __gettextinfo(edu_dict, edu_span):
        """ Get text span for parent node

        :type edu_dict: dict of list
        :param edu_dict: EDU from this document

        :type edu_span: tuple with two elements
        :param edu_span: start/end of EDU IN this span
        """
        # text = lnode.text + " " + rnode.text
        text = []
        for idx in range(edu_span[0], edu_span[1] + 1, 1):
            text += edu_dict[idx]
        # Return: A list of token indices
        return text

    @staticmethod
    def extract_relation(s, level=0):
        """ Extract discourse relation on different level
        """
        return rel2class[s]

    def bracketing(self):
        """ brackets according an Binary RST tree
        """
        nodelist = RstTree.postorder_DFT(self.tree, [])
        nodelist.pop()  # Remove the root node
        brackets = []
        for i, node in enumerate(nodelist):
            relation = RstTree.extract_relation(node.relation)
            b = (node.edu_span, node.prop, relation)
            brackets.append(b)
        return brackets

    # newly added for .rs3 conversion preparation
    def parse(self):
        """ Get parse tree in string format

            For visualization, use nltk.tree:
            from nltk.tree import Tree
            t = Tree.fromstring(parse)
            t.draw()
        """
        parse = getparse(self.tree, "")
        return parse

