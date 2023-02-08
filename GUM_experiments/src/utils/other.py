import torch


class ParseError(Exception):
    """ Exception for parsing
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ActionError(Exception):
    """ Exception for illegal parsing action
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def reverse_dict(dct):
    """ Reverse the {key:val} in dct to
        {val:key}
    """
    newmap = {}
    for (key, val) in dct.items():
        newmap[val] = key
    return newmap


def collate_samples(data_helper, sample_list):
    all_docs, all_indexes, all_action_feats, all_neural_feats, all_actions, all_relations = [], [], [], [], [], []
    for feats, action in sample_list:
        all_docs.append(data_helper.docs[feats[0]])
        all_indexes.append(feats[0])
        all_action_feats.append(feats[1][0])
        all_neural_feats.append(feats[1][1])
        all_actions.append(action[0])
        all_relations.append(action[1])
    # all_clusters = [data_helper.all_clusters[i] for i in all_indexes]
    all_clusters = []
    all_actions, all_relations = torch.stack(all_actions), torch.stack(all_relations)
    return all_docs, all_clusters, \
           all_action_feats, all_neural_feats, \
           all_actions, all_relations, \
           all_relations > 0


def cleanup_load_dict(model_save):
    if 'operational_feats.weight' in model_save['model_state_dict']:
        del model_save['model_state_dict']['operational_feats.weight']
        del model_save['model_state_dict']['t12_dep_type_feats.weight']
        del model_save['model_state_dict']['st_q_dep_type_feats.weight']
        del model_save['model_state_dict']['subtree_form_feats.weight']
        del model_save['model_state_dict']['edu_length_feats.weight']
        del model_save['model_state_dict']['sent_length_feats.weight']
        del model_save['model_state_dict']['edu_comp_feats.weight']
        del model_save['model_state_dict']['sent_comp_feats.weight']
        del model_save['model_state_dict']['stack_cat_feats.weight']
        del model_save['model_state_dict']['queue_cat_feats.weight']


# RST-DT Manual: classes p.32; all relations p.42
# RST-DT: Multinuclear Relations are in uppercase (the first letter):
# e.g. comparison (Mononuclear - satellite); Comparison (Multinuclear)

# updated on 2022/02/18: use GUM8 relation collapsed for prediction
class2rel = {
    'Attribution': ['attribution', 'attribution-e', 'attribution-n', 'attribution-negative', 'attribution-positive'],
    'Context': ['background', 'background-e', 'circumstance', 'circumstance-e', 'context-background', 'context-circumstance'],
    'Causal': ['cause', 'Cause-Result', 'result', 'result-e',
              'Consequence', 'consequence', 'consequence-n-e', 'consequence-n', 'consequence-s-e', 'consequence-s',
               'causal-cause', 'causal-result'],
    'Contingency': ['condition', 'condition-e', 'hypothetical', 'contingency', 'otherwise', 'Otherwise', 'contingency-condition'],
    'Adversative': ['Contrast', 'concession', 'concession-e', 'antithesis', 'antithesis-e',
                    'adversative-antithesis', 'adversative-concession', 'adversative-contrast'],
    'Elaboration': ['elaboration-additional-e', 'elaboration-general-specific',
                    'elaboration-general-specific-e', 'elaboration-part-whole', 'elaboration-part-whole-e',
                    'elaboration-process-step', 'elaboration-process-step-e', 'elaboration-object-attribute-e',
                    'elaboration-object-attribute', 'elaboration-set-member', 'elaboration-set-member-e', 'example',
                    'example-e', 'definition', 'definition-e', 'elaboration-attribute', 'elaboration-additional'],
    'Evaluation': ['evaluation', 'evaluation-n', 'evaluation-s-e', 'evaluation-s', 'Evaluation',
                   'interpretation', 'interpretation-n', 'interpretation-s-e', 'interpretation-s', 'Interpretation',
                   'conclusion', 'Conclusion', 'comment', 'comment-e', 'evaluation-comment'],
    'Explanation': ['evidence', 'evidence-e', 'explanation-argumentative', 'explanation-argumentative-e', 'Reason', 'reason',
                    'reason-e', "explanation-evidence", "explanation-justify", 'explanation-motivation'],
    'Joint': ['List', 'Disjunction', 'joint-disjunction', 'joint-list', 'joint-other', 'joint-sequence',
              'topic-shift', 'topic-drift', 'Topic-Shift', 'Topic-Drift', 
              'temporal-before', 'temporal-before-e', 'temporal-after', 'temporal-after-e', 'Temporal-Same-Time', 'temporal-same-time', 'temporal-same-time-e', 'Sequence', 'Inverted-Sequence', 
              'Comparison', 'comparison', 'comparison-e', 'preference', 'preference-e', 'Analogy', 'analogy', 'analogy-e', 'proportion', 'Proportion'],
    'Mode': ['manner', 'manner-e', 'means', 'means-e', 'mode-manner', 'mode-means'],
    'Organization': ["organization-heading", "organization-phatic", "organization-preparation", 'TextualOrganization'],
    'Purpose': ["purpose-attribute", "purpose-goal", 'purpose', 'purpose-e', 'enablement', 'enablement-e'],
    'Restatement': ["restatement-partial", "restatement-repetition", 'summary', 'summary-n', 'summary-s', 'restatement-e', 'restatement'],
    "Topic": ["topic-question", "topic-solutionhood",
              'problem-solution', 'problem-solution-n', 'problem-solution-s', 'Problem-Solution',
              'Question-Answer', 'question-answer', 'question-answer-n', 'question-answer-s',
              'Statement-Response', 'statement-response-n', 'statement-response-s',
              'Topic-Comment', 'Comment-Topic', 'rhetorical-question'],
    'span': ['span'],
    'Same-Unit': ['Same-Unit', 'same-unit'],  # same-unit - lowercase for GUM
}


# updated on Feb 18 2022: GUM 8 relation classes: 15: 1-15 + the RST-DT classes for RST-DT test: 17-18
relation_map = {None: 0, 'Adversative': 1, "Attribution": 2, 'Causal': 3, 'Context': 4, 'Contingency': 5,
                'Elaboration': 6, "Explanation": 7, 'Evaluation': 8, 'Joint': 9, "Mode": 10, 'Organization': 11,
                'Purpose': 12, 'Restatement': 13, 'Same-Unit': 14, 'Topic': 15}


action_map = {('Shift', None): 0, ('Reduce', 'NS'): 1, ('Reduce', 'NN'): 2, ('Reduce', 'SN'): 3}

xidx_action_map, xidx_relation_map = reverse_dict(action_map), reverse_dict(relation_map)

rel2class = {}
for cl, rels in class2rel.items():
    rel2class[cl] = cl
    for rel in rels:
        rel2class[rel] = cl

