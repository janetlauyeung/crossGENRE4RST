import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer

import os, io, re, attr, random
from collections import defaultdict
from fnmatch import fnmatch
from copy import deepcopy as c
import pickle

from ubc_coref.utils import *

NORMALIZE_DICT = {"/.": ".", "/?": "?",
                  "-LRB-": "(", "-RRB-": ")",
                  "-LCB-": "{", "-RCB-": "}",
                  "-LSB-": "[", "-RSB-": "]"}
REMOVED_CHAR = ["/", "%", "*"]

bert_tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
max_segment_len = 384

class Corpus:
    def __init__(self, documents):
        self.docs = documents

    def __getitem__(self, idx):
        return self.docs[idx]

    def __repr__(self):
        return 'Corpus containg %d documents' % len(self.docs)    

class Document:
    
    def __init__(self, raw_text, tokens, sents, corefs, speakers, genre, filename):
        self.raw_text = raw_text
        self.tokens = tokens
        self.sents = sents
        self.corefs = corefs
        self.speakers = speakers
        self.genre = genre
        self.filename = filename
        self.sents = sents
        self._compute_bert_tokenization()
        # Filled in at evaluation time.
        self.tags = None

    def __getitem__(self, idx):
        return (self.tokens[idx], self.corefs[idx], \
                self.speakers[idx], self.genre)

    def __repr__(self):
        return 'Document containing %d tokens' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)
        
    def _compute_bert_tokenization(self):
        
        self.bert_tokens = []
        self.word_subtoken_map = []
        self.word2idx = {}
        self.segment_subtoken_map = []
        self.segments = []
        
        self.sentence_ends, token_ends, self.sentence_ends_subtok = [], [], []
        idx_word = 0
        idx_subtoken = 0
        self.word2sent = []
        self.sent2subtok_bdry = []
        
        
        for j, sent in enumerate(self.sents):
            for i, token in enumerate(sent):
                
                subtokens = bert_tokenizer(token)['input_ids'][1:-1]
                
                self.bert_tokens.extend(subtokens)
                
                token_ends.extend([False] * len(subtokens))
                token_ends[-1] = True
                self.sentence_ends.extend([False] * len(subtokens))
                
                self.word_subtoken_map.extend([idx_word] * len(subtokens))
                self.word2sent.append(j)
                idx_word += 1
                idx_subtoken += len(subtokens)
                
            self.sentence_ends[-1] = True
        
        current = 0
        previous_token = 0
        
        while current < len(self.bert_tokens):
            # Min of last token of the document, or 
            # - 2 to include [CLS] and [SEP] tokens, -1 to refer to the corrent arr element
            end = min(current + max_segment_len - 1 - 2, len(self.bert_tokens) - 1)

            while end >= current and not self.sentence_ends[end]:
                # make end of segment be end of some sentence, or equal to current token
                end -= 1
            # How can end be less than current? Only if it is less by 1 (previous constraint not satisfied)
            if end < current:
                # Put the end token back?
                end = min(current + max_segment_len - 1 - 2,  len(self.bert_tokens) - 1)
                assert self.word2sent[self.word_subtoken_map[current]] == self.word2sent[self.word_subtoken_map[end]]
                # Make the end be end of last token
                while end >= current and not token_ends[end]:
                    end -= 1
                if end < current:
                    raise Exception("Can't find valid segment")
                    

            # Make segment consist of subtokens for found boundaries
            self.segments.append(torch.LongTensor([101] + self.bert_tokens[current : end + 1] + [102]))
            
            subtoken_map = self.word_subtoken_map[current : end + 1]
            
            # Make the [CLS] token of the segment map to last word of previous segment and [SEP] token
            # to last word in the current segment.
            self.segment_subtoken_map.extend([previous_token] + subtoken_map + [subtoken_map[-1]])
            
            subtoken_sent_ends = self.sentence_ends[current : end + 1]
            subtoken_sent_ends[-1] = False
            self.sentence_ends_subtok.extend([False] + subtoken_sent_ends + [True])
                
            current = end + 1
            previous_token = subtoken_map[-1]
            
            
        self.word2subtok = defaultdict(list)
        sentence_idx = 0
        for i, word_idx in enumerate(self.segment_subtoken_map):
            self.word2subtok[word_idx].append(i)
            # If current token is an end of sentence
            if self.sentence_ends_subtok[i]:
                self.sent2subtok_bdry.append((sentence_idx, i))
                sentence_idx = i+1
            

    
    def spans(self):
        """ Create Span object for each span """
        return [Span(i1=i[0], i2=i[-1], id=idx,
                    speaker=self.speaker(i), genre=self.genre)
                for idx, i in enumerate(compute_idx_spans(self.sents))]

    def truncate(self, use_bert, MAX=50):
        """ Randomly truncate the document to up to MAX sentences """
        if use_bert:
            if len(self.segments) > MAX:
                i = random.sample(range(MAX, len(self.segments)), 1)[0]
                subtokens = flatten(self.segments[i-MAX:i])
                
                
                # Index of the first token in the truncated segments
                num_pre_subtokens = len(flatten(self.segments[0:i-MAX]))
                # Index of the last token in the truncated segments
                num_pre_curr_subtokens = num_pre_subtokens + len(subtokens) -1
                
                # Index of the first and the last word corresponding to 
                # given truncated segments
                first_word_idx, last_word_idx = self.segment_subtoken_map[num_pre_subtokens], \
                                                    self.segment_subtoken_map[num_pre_curr_subtokens]
                
                first_sentence_idx, last_sentence_idx = self.word2sent[first_word_idx], \
                                                            self.word2sent[last_word_idx]
                sents = self.sents[first_sentence_idx:last_sentence_idx + 1]
                 # +1 to include last sentence too
                tokens = flatten(sents)
                pre_sents = self.sents[0:first_sentence_idx]
                pre_tokens = flatten(pre_sents)
                # Index of first token in truncated sentences
                num_pre_tokens = len(pre_tokens)
                # Index of last token in truncated sentences
                num_pre_curr_tokens = num_pre_tokens + len(tokens) - 1
                new_corefs = []

                for coref in self.corefs:
                    # Ignore corefs outside of current sentences
                    if coref['start'] < num_pre_tokens or coref['end'] > num_pre_curr_tokens:
                        continue
                    new_start = coref['start'] - num_pre_tokens
                    new_end = coref['end'] - num_pre_tokens
                    new_coref = {'label': coref['label'], 
                                 'start': new_start,
                                 'end': new_end,
                                 'span': (new_start, new_end)}
                    new_corefs.append(new_coref)

                new_speakers = self.speakers[num_pre_tokens:num_pre_curr_tokens + 1]

                return self.__class__(c(self.raw_text), tokens, sents,
                                      new_corefs, new_speakers,
                                      c(self.genre), c(self.filename))                
        else:    
            if len(self.sents) > MAX:
                # num_sents >= i >= MAX
                i = random.sample(range(MAX, len(self.sents)), 1)[0]
                tokens = flatten(self.sents[i-MAX:i])

                pre_sents = self.sents[0:i-MAX]
                pre_tokens = flatten(pre_sents)
                # Index of first token in truncated sentences
                num_pre_tokens = len(pre_tokens)
                # Index of last token in truncated sentences
                num_pre_curr_tokens = num_pre_tokens + len(tokens) - 1
                new_corefs = []

                for coref in self.corefs:
                    # Ignore corefs outside of current sentences
                    if coref['start'] < num_pre_tokens or coref['end'] > num_pre_curr_tokens:
                        continue
                    new_start = coref['start'] - num_pre_tokens
                    new_end = coref['end'] - num_pre_tokens
                    new_coref = {'label': coref['label'], 
                                 'start': new_start,
                                 'end': new_end,
                                 'span': (new_start, new_end)}
                    new_corefs.append(new_coref)

                new_speakers = self.speakers[num_pre_tokens:num_pre_curr_tokens + 1]

                return self.__class__(c(self.raw_text), tokens, self.sents[i-MAX:i],
                                      new_corefs, new_speakers,
                                      c(self.genre), c(self.filename))
        return self

    def speaker(self, i):
        """ Compute speaker of a span """
        if self.speakers[i[0]] == self.speakers[i[-1]]:
            return self.speakers[i[0]]
        return None
    
    
    def speaker_start_end(self, start, end):
        """ Compute speaker of a span """
        if self.speakers[start] == self.speakers[end]:
            return self.speakers[start]
        return None

@attr.s(frozen=True, repr=False)
class Span:

    # Left / right token indexes
    i1 = attr.ib()
    i2 = attr.ib()

    # Id within total spans (for indexing into a batch computation)
    id = attr.ib()

    # Speaker
    speaker = attr.ib()

    # Genre
    genre = attr.ib()

    # Unary mention score, as tensor
    si = attr.ib(default=None)

    # List of candidate antecedent spans
    yi = attr.ib(default=[])

    # Corresponding span ids to each yi
    yi_idx = attr.ib(default=None)

    def __len__(self):
        return self.i2-self.i1+1

    def __repr__(self):
        return 'Span representing %d tokens' % (self.__len__())

def read_corpus(dirname):
    conll_files = parse_filenames(dirname=dirname, pattern="*gold_conll")
    return Corpus(flatten([load_file(file) for file in conll_files]))

def load_file(filename):
    """ Load a *._conll file
    Input:
        filename: path to the file
    Output:
        documents: list of Document class for each document in the file containing:
            tokens:                   split list of text
            utts_corefs:
                coref['label']:     id of the coreference cluster
                coref['start']:     start index (index of first token in the utterance)
                coref['end':        end index (index of last token in the utterance)
                coref['span']:      corresponding span
            utts_speakers:          list of speakers
            genre:                  genre of input
    """
    documents = []
    with io.open(filename, 'rt', encoding='utf-8', errors='strict') as f:
        raw_text, tokens, sents, text, utts_corefs, utts_speakers, corefs, index = [], [], [], [], [], [], [], 0
        genre = filename.split('/')[6]
        for line in f:
            raw_text.append(line)
            cols = line.split()

            # End of utterance within a document: update lists, reset variables for next utterance.
            if len(cols) == 0:
                if text:
                    tokens.extend(text), sents.append(text)
                    utts_corefs.extend(corefs), utts_speakers.extend([speaker]*len(text))
                    text, corefs = [], []
                    continue

            # End of document: organize the data, append to output, reset variables for next document.
            elif len(cols) == 2:
                doc = Document(raw_text, tokens, sents, utts_corefs, utts_speakers, genre, filename)
                documents.append(doc)
                raw_text, tokens, sents, text, utts_corefs, utts_speakers, index = [], [], [], [], [], [], 0
 
            # Inside an utterance: grab text, speaker, coreference information.
            elif len(cols) > 7:
                text.append(clean_token(cols[3]))
                speaker = cols[9]

                # If the last column isn't a '-', there is a coreference link
                if cols[-1] != u'-':
                    coref_expr = cols[-1].split('|')
                    
                    for part in coref_expr:
                        if part[0] == "(":
                            if part[-1] == ")":
                                label = part[1:-1]
                                end = index
                            else:
                                label = part[1:]
                                end = None
                            corefs.append({'label': label,
                                           'start': index,
                                           'end': end,
                                           'span': (index, index)})
                        else:
                            label = part[:-1]
                            found = False
                            for i in range(len(corefs)-1, -1, -1):
                                if corefs[i]['label'] == label and corefs[i]['end'] is None:
                                    found = True
                                    break
                                    
                            if not found:
                                raise Exception("Coref not found")
                                    
                            corefs[i].update({'end': index,
                                              'span': (corefs[i]['start'], index)})
                    
                index += 1
            else:
                # Beginning of Document, beginning of file, end of file: nothing to scrape off
                continue

    return documents

def parse_filenames(dirname, pattern = "*conll"):
    """ Walk a nested directory to get all filename ending in a pattern """
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, pattern):
                yield os.path.join(path, name)

def clean_token(token):
    """ Substitute in /?(){}[] for equivalent CoNLL-2012 representations,
    remove /%* """
    cleaned_token = token
    if cleaned_token in NORMALIZE_DICT:
        cleaned_token = NORMALIZE_DICT[cleaned_token]

    if cleaned_token not in REMOVED_CHAR:
        for char in REMOVED_CHAR:
            cleaned_token = cleaned_token.replace(char, u'')

    if len(cleaned_token) == 0:
        cleaned_token = ","
    return cleaned_token

def lookup_tensor(tokens, vectorizer):
    """ Convert a sentence to an embedding lookup tensor """
    return to_cuda(torch.tensor([vectorizer.stoi(t) for t in tokens]))

def load_corpus_portion(corpus):
    if corpus not in ['train', 'val', 'test']:
        raise Exception("Unknown corpus type")
    full_path = '../data/' + corpus + '_corpus_' + str(max_segment_len) + '.pkl'
    if os.path.exists(full_path):
        corpus = pickle.load(open(full_path, 'rb'))
    else:
        corpus = read_corpus('../data/' + corpus if corpus != 'val' else 'development')
        pickle.dump(corpus, open(full_path, 'wb'))
    return corpus