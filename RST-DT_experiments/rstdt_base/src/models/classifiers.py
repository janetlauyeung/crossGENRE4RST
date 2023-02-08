from utils.constants import *
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from ubc_coref.utils import *
from utils.other import relation_map


class NeuralClassifier(nn.Module):
    
    def __init__(self, data_helper, config):
        super(NeuralClassifier, self).__init__()    
        
        self.config = config
        self.init_embeddings()
        
        # https://huggingface.co/SpanBERT/spanbert-base-cased
        self.tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
        self.bert = BertModel.from_pretrained("SpanBERT/spanbert-base-cased")
        
        self.coref_projection = nn.Sequential(
                nn.Linear(768*2, 768),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(768, 768)
        )
        self.coref_score = nn.Sequential(
                nn.Linear(768, 300),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(300, 1)
            )            
        self.out_classifier_action = nn.Sequential(
            nn.Linear(2444, self.config[HIDDEN_DIM]),  
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.config[HIDDEN_DIM], 4)
        )
        self.out_classifier_rel = nn.Sequential(
            nn.Linear(2444, self.config[HIDDEN_DIM]),  
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.config[HIDDEN_DIM], len(relation_map))
        )
            
    def init_embeddings(self):
        # Wang etal features, encode binary(True/False) or categorical features
        self.t12_same_sent_feats = nn.Embedding(2, 5)
        self.t12_same_para_feats = nn.Embedding(2, 5)
        self.st_q_same_sent_feats = nn.Embedding(2, 5)
        self.st_q_same_para_feats = nn.Embedding(2, 5)
        self.t12_q_same_sent_feats = nn.Embedding(2, 5)
        self.t12_q_same_para_feats = nn.Embedding(2, 5)

        self.top1_sent_start_feats = nn.Embedding(2, 5)
        self.top1_para_start_feats = nn.Embedding(2, 5)
        self.top1_doc_start_feats = nn.Embedding(2, 5)
        self.top1_sent_end_feats = nn.Embedding(2, 5)
        self.top1_para_end_feats = nn.Embedding(2, 5)
        self.top1_doc_end_feats = nn.Embedding(2, 5)

        self.top2_sent_start_feats = nn.Embedding(2, 5)
        self.top2_para_start_feats = nn.Embedding(2, 5)
        self.top2_doc_start_feats = nn.Embedding(2, 5)
        self.top2_sent_end_feats = nn.Embedding(2, 5)
        self.top2_para_end_feats = nn.Embedding(2, 5)
        self.top2_doc_end_feats = nn.Embedding(2, 5)

        self.queue1_sent_start_feats = nn.Embedding(2, 5)
        self.queue1_para_start_feats = nn.Embedding(2, 5)
        self.queue1_doc_start_feats = nn.Embedding(2, 5)
        self.queue1_sent_end_feats = nn.Embedding(2, 5)
        self.queue1_para_end_feats = nn.Embedding(2, 5)
        self.queue1_doc_end_feats = nn.Embedding(2, 5)
        
        self.pad_60 = torch.zeros((1, 60), device=self.config[DEVICE])
        self.pad_20 = torch.zeros((1, 20), device=self.config[DEVICE])
        # added
        self.pad_10 = torch.zeros((1, 10), device=self.config[DEVICE])

    def decode_action_coref(self, edu_embeds, action_feats):
        clf_feats = [[] for x in range(len(edu_embeds))]
        
        for i, node in enumerate(edu_embeds):
            clf_feats[i].append(node)
            clf_feats[i] = self.add_action_feats(clf_feats[i], action_feats[i]).unsqueeze(0)
            
        clf_feats = torch.cat(clf_feats, dim=0)
        
        return self.out_classifier_action(clf_feats), self.out_classifier_rel(clf_feats)
    
    def get_edus_bert_coref(self, docs, batch_clusters, batch_spans): 
        # For extracting initial BERT embeddings
        embed_module = self.bert.get_input_embeddings()

        # Batching stuff
        all_segs, all_masks, all_seg_lens, per_batch_segs = [], [], [], []

        # Preprocess each doc in the batch with coreference info
        for i, (doc, clusters) in enumerate(zip(docs, batch_clusters)):

            # Lengths per segment
            seg_lens = [len(s) for s in doc.segments]

            # Lengths per segment per batch
            all_seg_lens.append(seg_lens)

            # Segments per batch
            per_batch_segs.append(len(doc.segments))

            # Get initial BERT embeddings for each token in the document
            # Pad token ids to length of 384 with [PAD] tokens
            segments = pad_and_stack([seg.unsqueeze(1) 
                                      for seg in doc.segments], pad_size=384).squeeze(2).to(self.config[DEVICE])
            mask = segments > 0
            assert segments.shape[0] == len(seg_lens) and segments.shape[1] == 384

            # Initial SpanBERT embeddings (not contextualized)
            segments = embed_module(segments)

            # Only apply coref feats for SpanBERT-Coref model
            if self.config[MODEL_TYPE] == 1:
                # Remove [PAD]s, linearize
                flat_toks = unpad_toks(segments, mask)

                # Update token representations with coref info
                coref_aug_toks = self.apply_coref_clusters(flat_toks, doc, clusters)
                assert coref_aug_toks.shape == flat_toks.shape

                # Reshape back to num_seg sequences, padded with [PAD]
                segments = pad_and_stack(torch.split(coref_aug_toks, 
                                                    split_size_or_sections=seg_lens), 
                                         pad_size=384)
                assert segments.shape[0] == len(seg_lens) and segments.shape[1] == 384

            # Save for batching
            all_segs.append(segments)
            all_masks.append(mask)

        # Process segments for all batches in parallel
        all_segs, all_masks = torch.cat(all_segs),  torch.cat(all_masks)

        assert all_segs.dim() == 3 and all_segs.shape[0] == sum(per_batch_segs) and \
                all_segs.shape[1] == 384 and all_segs.shape[2] == 768

        # Contextualize the embeddings
        context_segments = self.bert(inputs_embeds=all_segs, attention_mask=all_masks.byte())[0]
        assert context_segments.shape == all_segs.shape

        # Extract per-document segments and masks
        all_context_segments = list(torch.split(context_segments, per_batch_segs, dim=0))

        all_masks = torch.split(all_masks, per_batch_segs, dim=0)

        # Get linearized token sequence for each document without [PAD]s
        for i, (context_segments, mask) in enumerate(zip(all_context_segments, all_masks)):
            all_context_segments[i] = unpad_toks(context_segments, mask)            

        # Extract nuclear EDU tokens
        batch_span_embeds = self.extract_nuclear_spans(all_context_segments, batch_spans, docs, batch_clusters)
        return batch_span_embeds
        
    def extract_nuclear_spans(self, doc_word_embeds, all_spans, docs, batch_clusters=None):
        
        batch_span_embeds = []
        for i, (doc_word_embed, const_spans, doc) in enumerate(zip(doc_word_embeds, all_spans, docs)):
            span_embeds = []
            sent2token, word2token = doc.sent2subtok_bdry, doc.word2subtok
            
            # If top queue element is missing:
            if const_spans[0][0] != "QueueEDUs1":
                span_embeds.append(torch.zeros(768).to(self.config[DEVICE]))

            # Extract span summaries 
            for span in const_spans:
                # nuc_edu = span[2]
                nuc_edu = span[1]
                # EDU ids start with 1
                edu_idx = nuc_edu[0] - 1
                span_start, span_end = sent2token[edu_idx]
                                           
                start_emb, end_emb = doc_word_embed[span_start], doc_word_embed[span_end]
                
                span_embeds.append((start_emb + end_emb) / 2)

            # Pad with zeros if stack is empty/has 1 element
            while len(span_embeds) < 3:
                span_embeds.append(torch.zeros(768).to(self.config[DEVICE]))
            
            # A single vector from all span embeddings
            span_embeds = torch.cat(span_embeds)
            assert span_embeds.dim() == 1 and span_embeds.shape[0] == 768 * 3
            batch_span_embeds.append(span_embeds)
            
        return batch_span_embeds

    def apply_coref_clusters(self, tokens, doc, clusters):
        # For Coref-Feats model
        for cluster in clusters:
            word2tok = doc.word2subtok
            tokens = tokens.clone()
            cluster_acc = []
            for mention in cluster:
                cluster_acc.append(torch.arange(start=word2tok[mention[0]][0], 
                                                end=word2tok[mention[1]][-1] + 1, 
                                                device=self.config[DEVICE]))
            # Select tokens in the cluster
            cluster_acc = torch.unique(torch.cat(cluster_acc, dim=0), sorted=False) 
            cluster_toks = tokens[cluster_acc]
            
            # Compute the cluster representation
            cluster_tok_att = softmax(self.coref_score(cluster_toks), dim=0)
            clust_repres = torch.sum(cluster_toks * cluster_tok_att, dim=0)
            clust_repres = clust_repres.unsqueeze(0).repeat(cluster_toks.shape[0], 1)
                        
            # Update cluster tokens
            clust_aug_toks = torch.cat([cluster_toks, clust_repres], dim=1)
            f_n = torch.sigmoid(self.coref_projection(clust_aug_toks))
            tokens[cluster_acc] = f_n * cluster_toks + (1 - f_n) * clust_repres
        
        return tokens            

    def add_action_feats(self, node_edus, action_feats):
        feat_vector = []
        trueTensor, falseTensor = torch.LongTensor([1]).to(self.config[DEVICE]), torch.LongTensor([0]).to(self.config[DEVICE])

        for i, feature in enumerate(action_feats):
            if len(feature) == 3:
                if feature[2] == 1:
                    query = trueTensor
                else:
                    query = falseTensor
                    
            if feature[0] == TOP12_STACK:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_20)
                elif feature[1] in [SAME_SENT, SENT_CONTINUE]:
                    feat_vector.append(self.t12_same_sent_feats(query))
                elif feature[1] in [SAME_PARA, PARA_CONTINUE]:
                    feat_vector.append(self.t12_same_para_feats(query))
                else:
                    raise ValueError("Unrecognized feature: ", feature)

            elif feature[0] == STACK_QUEUE:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_20)
                elif feature[1] in [SAME_SENT, SENT_CONTINUE]:
                    feat_vector.append(self.st_q_same_sent_feats(query))
                elif feature[1] in [SAME_PARA, PARA_CONTINUE]:
                    feat_vector.append(self.st_q_same_para_feats(query))
                else:
                    raise ValueError("Unrecognized feature: ", feature)                    

            elif feature[0] == TOP12_STACK_QUEUE:
                if feature[1] == SAME_SENT:
                    feat_vector.append(self.t12_q_same_sent_feats(query))
                elif feature[1] == SAME_PARA:
                    feat_vector.append(self.t12_q_same_para_feats(query))
                else:
                    raise ValueError("Unrecognized feature: ", feature)

            elif feature[0] == TOP_1:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_10)
                elif feature[1] == SENT_START:
                    feat_vector.append(self.top1_sent_start_feats(query))
                elif feature[1] == SENT_END:
                    feat_vector.append(self.top1_sent_end_feats(query))
                elif feature[1] == PARA_START:
                    feat_vector.append(self.top1_para_start_feats(query))
                elif feature[1] == PARA_END:
                    feat_vector.append(self.top1_para_end_feats(query))
                elif feature[1] == DOC_START:
                    feat_vector.append(self.top1_doc_start_feats(query))
                elif feature[1] == DOC_END:
                    feat_vector.append(self.top1_doc_end_feats(query))             
                else:
                    raise ValueError("Unrecognized feature: ", feature)

            elif feature[0] == TOP_2:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_10)
                elif feature[1] == SENT_START:
                    feat_vector.append(self.top2_sent_start_feats(query))
                elif feature[1] == SENT_END:
                    feat_vector.append(self.top2_sent_end_feats(query))
                elif feature[1] == PARA_START:
                    feat_vector.append(self.top2_para_start_feats(query))
                elif feature[1] == PARA_END:
                    feat_vector.append(self.top2_para_end_feats(query))
                elif feature[1] == DOC_START:
                    feat_vector.append(self.top2_doc_start_feats(query))
                elif feature[1] == DOC_END:
                    feat_vector.append(self.top2_doc_end_feats(query))             
                else:
                    raise ValueError("Unrecognized feature: ", feature)

            elif feature[0] == QUEUE_1:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_10)
                elif feature[1] == SENT_START:
                    feat_vector.append(self.queue1_sent_start_feats(query))
                elif feature[1] == SENT_END:
                    feat_vector.append(self.queue1_sent_end_feats(query))
                elif feature[1] == PARA_START:
                    feat_vector.append(self.queue1_para_start_feats(query))
                elif feature[1] == PARA_END:
                    feat_vector.append(self.queue1_para_end_feats(query))
                elif feature[1] == DOC_START:
                    feat_vector.append(self.queue1_doc_start_feats(query))
                elif feature[1] == DOC_END:
                    feat_vector.append(self.queue1_doc_end_feats(query))             
                else:
                    raise ValueError("Unrecognized feature: ", feature)
            else:
                raise ValueError("Unrecognized feature: ", feature)
                
        if feat_vector:
            node_edus = [node_edu.unsqueeze(0) for node_edu in node_edus]
            feat_vector.extend(node_edus)
            node_edus = torch.cat(feat_vector, dim=1).squeeze(0)
        else:
            node_edus = torch.cat(node_edus)
            
        return node_edus


def unpad_toks(tokens, mask):
    tokens = tokens.view(-1, 768)
    mask = mask.view(-1)
    return tokens[mask]
