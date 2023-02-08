import torch
import torch.nn as nn

from ubc_coref.model_building_blocks import BertDocumentEncoder, MentionScore, HigherOrderScore


class CorefScore(nn.Module):
    """ Super class to compute coreference links between spans
    """
    def __init__(self, distribute_model=False,
                       distance_dim=20,
                       genre_dim=20,
                       speaker_dim=20):

        super().__init__()
        
        # Initialize modules
        self.encoder = BertDocumentEncoder(distribute_model)
        attn_dim = 768
        embeds_dim = 768

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = attn_dim*2 + embeds_dim + distance_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim*3 + distance_dim + genre_dim + speaker_dim
        
        self.score_spans = MentionScore(gi_dim, attn_dim, distance_dim, L=30, 
                                        distribute_model=distribute_model)

        self.score_pairs = HigherOrderScore(gij_dim, 
                                            gi_dim, distance_dim, genre_dim, 
                                            speaker_dim, K=50, N=2)

    def forward(self, doc):
        """ Encode document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores..
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(doc)
        
        # Get mention scores for each span, prune
        spans, g_i, mention_scores = self.score_spans(states, embeds, doc)
        # If the document is too short
        if len(spans) <= 2:
            return None, None
        # Get pairwise scores for each span combo
        spans, coref_scores = self.score_pairs(spans, g_i, mention_scores)

        return spans, coref_scores, embeds
