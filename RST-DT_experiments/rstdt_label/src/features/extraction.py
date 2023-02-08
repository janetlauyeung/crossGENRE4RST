import torch
from utils.constants import *


class ActionFeatureGenerator(object):
    def __init__(self, stack, queue, action_hist, doc, data_helper, config):
        """ Initialization of features generator

        :type stack: list
        :param stack: list of Node instance

        :type queue: list
        :param queue: list of Node instance

        :type doc: Doc instance
        :param doc:
        """
        # -------------------------------------
        self.action_hist = action_hist
        self.doc = doc
        self.stack = stack
        self.queue = queue
        self.data_helper = data_helper
        self.config = config
        
        # Stack
        if len(stack) >= 2:
            self.top1span, self.top2span = stack[-1], stack[-2]
        elif len(stack) == 1:
            self.top1span, self.top2span = stack[-1], None
        else:
            self.top1span, self.top2span = None, None
            
        # Queue
        if len(queue) > 0:
            self.firstspan = queue[0]
        else:
            self.firstspan = None

    def gen_features(self):
        """ Queue EDU + 2 subtrees from the top of the stack
        """
        feat_list = []
        neural_feats = []
        # Textual organization features
        if self.config[ORG_FEATS]:        
            for feat in self.organizational_features():
                feat_list.append(feat)

        # predicted label (from a GUM-trained model) feature
        if self.config[NUC_FEATS]:
            for feat in self.nucleus_features():
                feat_list.append(feat)
                
        if self.config[DO_COREF]:
            if (self.firstspan is not None):
                neural_feats.append(("QueueEDUs1", self.firstspan.edu_span, self.firstspan.edu_span))
            if (self.top1span is not None):
                neural_feats.append(("StackEDUs1", self.top1span.edu_span, self.top1span.nuc_span))
            if (self.top2span is not None):
                neural_feats.append(("StackEDUs2", self.top2span.edu_span, self.top2span.nuc_span))
        else:
            if (self.firstspan is not None):
                neural_feats.append(("QueueEDUs1", self.firstspan.edu_span))
            if (self.top1span is not None):
                neural_feats.append(("StackEDUs1", self.top1span.edu_span))
            if (self.top2span is not None):
                neural_feats.append(("StackEDUs2", self.top2span.edu_span))

        return feat_list, neural_feats

    def nucleus_features(self):
        """ Feature extract from one single nucleus EDU """
        # https://pytorch.org/docs/stable/tensors.html

        # GUM-v8-dev labels (2022)
        seq_label_dict = {'ROOT': 0, 'adversative-antithesis': 1, 'adversative-concession': 2, 'adversative-contrast': 3,
                'attribution-negative': 4, 'attribution-positive': 5, 'causal-cause': 6, 'causal-result': 7,
                'context-background': 8, 'context-circumstance': 9, 'contingency-condition': 10,
                'elaboration-additional': 11, 'elaboration-attribute': 12, 'evaluation-comment': 13,
                'explanation-evidence': 14, 'explanation-justify': 15, 'explanation-motivation': 16,
                'joint-disjunction': 17, 'joint-list': 18, 'joint-other': 19, 'joint-sequence': 20,
                'mode-manner': 21, 'mode-means': 22, 'organization-heading': 23, 'organization-phatic': 24,
                'organization-preparation': 25, 'purpose-attribute': 26, 'purpose-goal': 27,
                'restatement-partial': 28, 'restatement-repetition': 29, 'same-unit': 30, 'topic-question': 31,
                'topic-solutionhood': 32}

        if self.top1span is not None:
            text1 = self.top1span.text      # text1: [155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166]
            eduidx = self.doc.token_dict[text1[0]].eduidx
            # obtain the seq_label of a given edu
            # (i.e. obtain the edu via eduidx from the edu_dict and locate the first token in that edu to get its label)
            some_tok = self.doc.token_dict[self.doc.edu_dict[eduidx][0]]        # e.g. some_tok >> totaling
            catTensor = torch.LongTensor([seq_label_dict[some_tok.seq_label]])  # e.g. tensor([17]); seq_label >> elaboration
            yield (TOP_1, NUCLEUS_FEAT, catTensor)
        else:
            yield (TOP_1, NOT_PRESENT)

        if self.firstspan is not None:
            text3 = self.firstspan.text
            eduidx = self.doc.token_dict[text3[0]].eduidx
            # obtain the seq_label of a given edu
            # (i.e. obtain the edu via eduidx from the edu_dict and locate the first token in that edu to get its label)
            some_tok = self.doc.token_dict[self.doc.edu_dict[eduidx][0]]
            catTensor = torch.LongTensor([seq_label_dict[some_tok.seq_label]])
            yield (QUEUE_1, NUCLEUS_FEAT, catTensor)
        else:
            yield (QUEUE_1, NOT_PRESENT)

    def organizational_features(self):
        trueTensor, falseTensor = torch.LongTensor([1]), torch.LongTensor([0]) 
        # ---------------------------------------
        # Whether within same sentence and paragraph
        # Span 1 and 2
        if self.top1span is not None and self.top2span is not None:
            text1, text2 = self.top1span.text, self.top2span.text
            if self.doc.token_dict[text2[-1]].sidx == self.doc.token_dict[text1[0]].sidx:
                yield (TOP12_STACK, SENT_CONTINUE, trueTensor)
            else:
                yield (TOP12_STACK, SENT_CONTINUE, falseTensor)
            if self.doc.token_dict[text2[-1]].pidx == self.doc.token_dict[text1[0]].pidx:
                yield (TOP12_STACK, PARA_CONTINUE, trueTensor)
            else:
                yield (TOP12_STACK, PARA_CONTINUE, falseTensor)
        else:
            yield (TOP12_STACK, NOT_PRESENT)

        # Span 1 and top span
        # First word from span 1, last word from span 3
        if self.top1span is not None and self.firstspan is not None:
            text1, text3 = self.top1span.text, self.firstspan.text
            if self.doc.token_dict[text1[-1]].sidx == self.doc.token_dict[text3[0]].sidx:
                yield (STACK_QUEUE, SENT_CONTINUE, trueTensor)
            else:
                yield (STACK_QUEUE, SENT_CONTINUE, falseTensor)
            if self.doc.token_dict[text1[-1]].pidx == self.doc.token_dict[text3[0]].pidx:
                yield (STACK_QUEUE, PARA_CONTINUE, trueTensor)
            else:
                yield (STACK_QUEUE, PARA_CONTINUE, falseTensor)
        else:
            yield (STACK_QUEUE, NOT_PRESENT)

        # # Last word from span 1, first word from span 2
        top12_stack_same_sent, top12_stack_same_para = False, False
        if self.top1span is not None and self.top2span is not None:
            text1, text2 = self.top1span.text, self.top2span.text
            if self.doc.token_dict[text1[-1]].sidx == self.doc.token_dict[text2[0]].sidx:
                top12_stack_same_sent = True
                yield (TOP12_STACK, SAME_SENT, trueTensor)
            else:
                yield (TOP12_STACK, SAME_SENT, falseTensor)
            if self.doc.token_dict[text1[-1]].pidx == self.doc.token_dict[text2[0]].pidx:
                top12_stack_same_para = True
                yield (TOP12_STACK, SAME_PARA, trueTensor)
            else:
                yield (TOP12_STACK, SAME_PARA, falseTensor)
        else:
             yield (TOP12_STACK, NOT_PRESENT)

        # # Span 1 and top span
        # # First word from span 1, last word from span 3
        stack_queue_same_sent, stack_queue_same_para = False, False
        if self.top1span is not None and self.firstspan is not None:
            text1, text3 = self.top1span.text, self.firstspan.text
            if self.doc.token_dict[text1[0]].sidx == self.doc.token_dict[text3[-1]].sidx:
                stack_queue_same_sent = True
                yield (STACK_QUEUE, SAME_SENT, trueTensor)
            else:
                yield (STACK_QUEUE, SAME_SENT, falseTensor)
            if self.doc.token_dict[text1[0]].pidx == self.doc.token_dict[text3[-1]].pidx:
                stack_queue_same_para = True
                yield (STACK_QUEUE, SAME_PARA, trueTensor)
            else:
                yield (STACK_QUEUE, SAME_PARA, falseTensor)
        else:
             yield (STACK_QUEUE, NOT_PRESENT)

        if top12_stack_same_sent and stack_queue_same_sent:
            yield (TOP12_STACK_QUEUE, SAME_SENT, trueTensor)
        else:
            yield (TOP12_STACK_QUEUE, SAME_SENT, falseTensor)
            
        if top12_stack_same_para and stack_queue_same_para:
            yield (TOP12_STACK_QUEUE, SAME_PARA, trueTensor)
        else:
            yield (TOP12_STACK_QUEUE, SAME_PARA, falseTensor)
        
        # ---------------------------------------
        # whether span is the start or end of sentence, paragraph or document
        if self.top1span is not None:
            text = self.top1span.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield (TOP_1, SENT_START, trueTensor)
            else:
                yield (TOP_1, SENT_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield (TOP_1, SENT_END, trueTensor)
            else:
                yield (TOP_1, SENT_END, falseTensor)
                
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield (TOP_1, PARA_START, trueTensor)
            else:
                yield (TOP_1, PARA_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield (TOP_1, PARA_END, trueTensor)
            else:
                yield (TOP_1, PARA_END, falseTensor)
                
            if text[0] - 1 < 0:
                yield (TOP_1, DOC_START, trueTensor)
            else:
                yield (TOP_1, DOC_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield (TOP_1, DOC_END, trueTensor)
            else:
                yield (TOP_1, DOC_END, falseTensor)
                
        else:
            yield (TOP_1, NOT_PRESENT)
            
        if self.top2span is not None:
            text = self.top2span.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield (TOP_2, SENT_START, trueTensor)
            else:
                yield (TOP_2, SENT_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield (TOP_2, SENT_END, trueTensor)
            else:
                yield (TOP_2, SENT_END, falseTensor)
                
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield (TOP_2, PARA_START, trueTensor)
            else:
                yield (TOP_2, PARA_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield (TOP_2, PARA_END, trueTensor)
            else:
                yield (TOP_2, PARA_END, falseTensor)
                
            if text[0] - 1 < 0:
                yield (TOP_2, DOC_START, trueTensor)
            else:
                yield (TOP_2, DOC_START, falseTensor)
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield (TOP_2, DOC_END, trueTensor)
            else:
                yield (TOP_2, DOC_END, falseTensor)
        else:
            yield (TOP_2, NOT_PRESENT)

        if self.firstspan is not None:
            text = self.firstspan.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield (QUEUE_1, SENT_START, trueTensor)
            else:
                yield (QUEUE_1, SENT_START, falseTensor)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield (QUEUE_1, SENT_END, trueTensor)
            else:
                yield (QUEUE_1, SENT_END, falseTensor)
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield (QUEUE_1, PARA_START, trueTensor)
            else:
                yield (QUEUE_1, PARA_START, falseTensor)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield (QUEUE_1, PARA_END, trueTensor)
            else:
                yield (QUEUE_1, PARA_END, falseTensor)
            if text[0] - 1 < 0:
                yield (QUEUE_1, DOC_START, trueTensor)
            else:
                yield (QUEUE_1, DOC_START, falseTensor)
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield (QUEUE_1, DOC_END, trueTensor)
            else:
                yield (QUEUE_1, DOC_END, falseTensor)
        else:
            yield (QUEUE_1, NOT_PRESENT)
