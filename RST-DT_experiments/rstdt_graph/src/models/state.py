from utils.document import Doc
from utils.other import ActionError, ParseError
from utils.span import SpanNode
import torch
from utils.constants import *


class ParsingState(object):
    def __init__(self, stack, queue, config=None):
        """ Initialization
        """
        self.Stack = stack
        self.Queue = queue
        self.config = config

    def init(self, doc):
        """ Using text to initialize Queue

        :type doc: Doc instance
        :param doc:
        """
        N = len(doc.edu_dict)
        for idx in range(1, N + 1, 1):
            node = SpanNode(prop=None)
            node.text = doc.edu_dict[idx]
            node.edu_span, node.nuc_span = (idx, idx), (idx, idx)
            node.nuc_edu = idx
            self.Queue.append(node)

    def operate(self, action_tuple, edu_embeddings=None, nuc_edus=None):
        """ According to parsing label to modify the status of
            the Stack/Queue
        """
        action, form, relation = action_tuple
        
        if action == 'Shift':
            if len(self.Queue) == 0:
                raise ActionError("Shift action error")
            node = self.Queue.pop(0)
            self.Stack.append(node)
        elif action == 'Reduce':
            if len(self.Stack) < 2:
                raise ActionError("Reduce action error")
            rnode = self.Stack.pop()
            lnode = self.Stack.pop()
            # Create a new node
            # Assign a value to prop, only when it is someone's
            # children node
            node = SpanNode(prop=None)
            # Children node
            node.lnode, node.rnode = lnode, rnode
            # Parent node of children nodes
            node.lnode.pnode, node.rnode.pnode = node, node
            # Node text: concatenate two word lists
            node.text = lnode.text + rnode.text
            # EDU span
            node.edu_span = (lnode.edu_span[0], rnode.edu_span[1])
            # Nuc span / Nuc EDU
            node.form = form
            if form == 'NN':
                node.nuc_span = (lnode.nuc_edu, lnode.nuc_edu)
                node.nuc_edu = lnode.nuc_edu
                node.lnode.prop = "Nucleus"
                node.rnode.prop = "Nucleus"
                node.lnode.relation = relation
                node.rnode.relation = relation
            elif form == 'NS':
                node.nuc_span = (lnode.nuc_edu, lnode.nuc_edu)
                node.nuc_edu = lnode.nuc_edu
                node.lnode.prop = "Nucleus"
                node.rnode.prop = "Satellite"
                node.rnode.relation = relation
                node.lnode.relation = "span"
            elif form == 'SN':
                node.nuc_span = (rnode.nuc_edu, rnode.nuc_edu)
                node.nuc_edu = rnode.nuc_edu
                node.lnode.prop = "Satellite"
                node.rnode.prop = "Nucleus"
                node.lnode.relation = relation
                node.rnode.relation = "span"
            else:
                raise ValueError("Unrecognized form: {}".format(form))
                
            self.Stack.append(node)
        else:
            raise ValueError("Unrecognized parsing action: {}".format(action))
            
    def is_action_allowed(self, action_tuple, doc):
        
        action, _, _ = action_tuple
        if self.config[KEEP_BOUNDARIES]:
            if action == 'Shift' and len(self.Stack) >= 2 and len(self.Queue) > 0:
                first_span, second_span, first_queue = self.Stack[-1].text, self.Stack[-2].text, self.Queue[0].text
                first_span_sidx = doc.token_dict[first_span[0]].sidx
                if first_span_sidx == doc.token_dict[second_span[0]].sidx and \
                                first_span_sidx < doc.token_dict[first_queue[0]].sidx:
                    return False
                first_span_pidx = doc.token_dict[first_span[0]].pidx
                if first_span_pidx == doc.token_dict[second_span[0]].pidx and \
                        first_span_pidx < doc.token_dict[first_queue[0]].pidx:
                    return False
        
        if action == 'Shift' and len(self.Queue) == 0:
            return False
        if action == 'Reduce' and len(self.Stack) < 2:
            return False
        return True

    def get_status(self):
        """ Return the status of the Queue/Stack
        """
        return self.Stack, self.Queue

    def end_parsing(self):
        """ Whether we should end parsing
        """
        if (len(self.Stack) == 1) and (len(self.Queue) == 0):
            return True
        elif (len(self.Stack) == 0) and (len(self.Queue) == 0):
            raise ParseError("Illegal stack/queue status")
        else:
            return False

    def get_parse_tree(self):
        """ Get the entire parsing tree
        """
        if (len(self.Stack) == 1) and (len(self.Queue) == 0):
            return self.Stack[0]
        else:
            return None
