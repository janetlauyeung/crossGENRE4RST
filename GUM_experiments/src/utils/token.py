
class Token(object):
    """ Token class
    """

    def __init__(self):
        # Paragraph index, Sentence index, token index (within sent)
        self.pidx, self.sidx, self.tidx = None, None, None
        # Word, Lemma
        self.word, self.lemma = None, None
        # POS tag
        self.pos = None
        # Dependency label, head index
        self.dep_label, self.hidx = None, None
        # NER, Partial parse tree
        self.ner, self.partial_parse = None, None
        # EDU index
        self.eduidx = None
