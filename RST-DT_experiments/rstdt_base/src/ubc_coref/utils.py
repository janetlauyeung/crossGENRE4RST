import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence

import numpy as np
from boltons.iterutils import pairwise, windowed
from itertools import groupby, combinations
from collections import defaultdict


def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x, cuda_id=0):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda(cuda_id)
    return x

def unpack_and_unpad(lstm_out, reorder):
    """ Given a padded and packed sequence and its reordering indexes,
    unpack and unpad it. Inverse of pad_and_pack """

    # Restore a packed sequence to its padded version
    unpacked, sizes = pad_packed_sequence(lstm_out, batch_first=True)

    # Restored a packed sequence to its original, unequal sized tensors
    unpadded = [unpacked[idx][:val] for idx, val in enumerate(sizes)]

    # Restore original ordering
    regrouped = [unpadded[idx] for idx in reorder]

    return regrouped

def pad_and_stack(tensors, pad_size=None, value=0):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    # Get their original sizes (measured in number of tokens)
    sizes = [s.shape[0] for s in tensors]

    # Pad size will be the max of the sizes
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    # TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
    padded = torch.stack([F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)], dim=0)

    return padded

def pack(tensors):
    """ Pack list of tensors, provide reorder indexes """

    # Get sizes
    sizes = [t.shape[0] for t in tensors]

    # Get indexes for sorted sizes (largest to smallest)
    size_sort = np.argsort(sizes)[::-1]

    # Resort the tensor accordingly
    sorted_tensors = [tensors[i] for i in size_sort]

    # Resort sizes in descending order
    sizes = sorted(sizes, reverse=True)

    # Pack the padded sequences
    packed = pack_sequence(sorted_tensors)

    # Regroup indexes for restoring tensor to its original order
    reorder = torch.tensor(np.argsort(size_sort), requires_grad=False)

    return packed, reorder


def prune(mention_scores, start_words, end_words, T, LAMBDA=0.4):
    #Prune mention scores to the top lambda percent.
    #Returns list of tuple(scores, indices, g_i) 

    # Only take top λT spans, where T = len(doc)
    STOP = int(LAMBDA * T)
    # Sort by mention score, remove overlapping spans, prune to top λT spans
    indices = torch.argsort(mention_scores.view(-1), descending=True)
    nonoverlapping_indices = remove_overlapping(torch.tensor(start_words)[indices], 
                                                torch.tensor(end_words)[indices], 
                                                indices, 
                                                STOP)
    
    # Resort by start, end indexes
    indices_sorted = sorted(nonoverlapping_indices, key=lambda i: (start_words[i], end_words[i]))

    return indices_sorted


def remove_overlapping(start_words, end_words, indices, STOP):
    #Remove spans that are overlapping by order of decreasing mention score
    #unless the current span i yields true to the following condition with any
    #previously accepted span j:

    #si.i1 < sj.i1 <= si.i2 < sj.i2   OR
    #sj.i1 < si.i1 <= sj.i2 < si.i2 
    
    # TODO: Pretty brute force (O(n^2)), rewrite it later
    nonoverlapping_indices, overlapped = [], False
    for i in range(len(start_words)):        
        for j in nonoverlapping_indices:
            if (start_words[i] < start_words[j] and start_words[j] <= end_words[i] and end_words[i] < end_words[j] or \
                    start_words[j] < start_words[i] and start_words[i] <= end_words[j] and end_words[j] < end_words[i]):
                
                overlapped = True
                break
                
        if not overlapped:
            nonoverlapping_indices.append(i)
                
        overlapped = False
                
        if len(nonoverlapping_indices) == STOP:
            break
        
    return indices[nonoverlapping_indices]

    
def pairwise_indexes(spans):
    """ Get indices for indexing into pairwise_scores """
    indexes = [0] + [len(s.yi) for s in spans]
    indexes = [sum(indexes[:idx+1]) for idx, _ in enumerate(indexes)]
    return pairwise(indexes)


def extract_gold_corefs(document):
    """ Parse coreference dictionary of a document to get coref links """

    # Initialize defaultdict for keeping track of corefs
    gold_links = defaultdict(list)

    # Compute number of mentions
    gold_mentions = set([coref['span'] for coref in document.corefs])
    total_mentions = len(gold_mentions)
    # Compute number of coreferences
    for coref_entry in document.corefs:

        # Parse label of coref span, the span itself
        label, span_idx = coref_entry['label'], coref_entry['span']

        # All spans corresponding to the same label
        gold_links[label].append(span_idx) # get all spans corresponding to some label
    # Flatten all possible corefs, sort, get number
    gold_corefs = flatten([[coref
                            for coref in combinations(gold, 2)]
                            for gold in gold_links.values()])
    
    gold_corefs = sorted(gold_corefs)
    total_corefs = len(gold_corefs)
    
    return gold_corefs, total_corefs, gold_mentions, total_mentions


def compute_idx_spans(sentences, L=10, word2tokens=None):
    # Compute span indexes for all possible spans up to length L in each
    #sentence 
    shift = 0    
    start_words, end_words, start_toks, end_toks, tok_ranges, word_widths, tok_widths = [], [], [], [], [], [], []
    for sent in sentences:
        sent_spans = []
        for length in range(1, min(L, len(sent))):
            l_spans = windowed(range(shift, len(sent)+shift), length)
            flattened = flatten_word2tokens(l_spans, word2tokens)     
            start_words.extend(flattened[0])
            end_words.extend(flattened[1])
            start_toks.extend(flattened[2])
            end_toks.extend(flattened[3])
            tok_ranges.extend(flattened[4])
            word_widths.extend(flattened[5])
            tok_widths.extend(flattened[6])
        shift += len(sent)
    return start_words, end_words, start_toks, end_toks, tok_ranges, word_widths, tok_widths


def s_to_speaker(span, speakers):
    """ Compute speaker of a span """
    if speakers[span.i1] == speakers[span.i2]:
        return speakers[span.i1]
    return None

def speaker_label(s1, s2):
    """ Compute if two spans have the same speaker or not """
    # Same speaker
    if s1.speaker == s2.speaker:
        idx = torch.tensor(1)

    # Different speakers
    elif s1.speaker != s2.speaker:
        idx = torch.tensor(2)

    # No speaker
    else:
        idx = torch.tensor(0)

    return to_cuda(idx)

def safe_divide(x, y):
    """ Make sure we don't divide by 0 """
    if y != 0:
        return x / y
    return 0

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]

def flatten_word2tokens(alist, word2tokens):
    """ Flatten a list of lists into one list """
    start_words, end_words, start_toks, \
            end_toks, tok_ranges, word_widths, \
            tok_widths = zip(*[(window[0], 
                                window[-1], 
                                word2tokens[window[0]][0], 
                                word2tokens[window[-1]][-1],
                                list(range(word2tokens[window[0]][0], word2tokens[window[-1]][-1] + 1)),
                                window[-1] - window[0] + 1,
                                word2tokens[window[-1]][-1] + 1 - word2tokens[window[0]][0])
                            for window in alist])
    
    return  start_words, end_words, start_toks, \
            end_toks, flatten(tok_ranges), word_widths, tok_widths
                