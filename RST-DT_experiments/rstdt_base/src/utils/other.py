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
# updated on Feb 6: added GUM8 relations to their corresponding RST-DT relation class for testing purposes
class2rel = {
    'Attribution': ['attribution', 'attribution-e', 'attribution-n', 'attribution-negative', 'attribution-positive'],
    'Background': ['background', 'background-e', 'circumstance', 'circumstance-e', 'context-background', 'context-circumstance'],
    'Cause': ['cause', 'Cause-Result', 'result', 'result-e',
              'Consequence', 'consequence', 'consequence-n-e', 'consequence-n', 'consequence-s-e', 'consequence-s', 'causal-cause', 'causal-result'],
    'Comparison': ['Comparison', 'comparison', 'comparison-e', 'preference', 'preference-e',
                   'Analogy', 'analogy', 'analogy-e', 'proportion', 'Proportion'],
    'Condition': ['condition', 'condition-e', 'hypothetical', 'contingency', 'otherwise', 'Otherwise', 'contingency-condition'],
    'Contrast': ['Contrast', 'concession', 'concession-e', 'antithesis', 'antithesis-e', 'adversative-antithesis', 'adversative-concession', 'adversative-contrast'],  # lowercase contrast: GUM relation
    'Elaboration': ['elaboration-additional', 'elaboration-additional-e', 'elaboration-general-specific',
                    'elaboration-general-specific-e', 'elaboration-part-whole', 'elaboration-part-whole-e',
                    'elaboration-process-step', 'elaboration-process-step-e', 'elaboration-object-attribute-e',
                    'elaboration-object-attribute', 'elaboration-set-member', 'elaboration-set-member-e', 'example',
                    'example-e', 'definition', 'definition-e', 'elaboration-attribute', 'elaboration-additional', 'purpose-attribute'],  # elaboration: GUM relation
    'Enablement': ['purpose', 'purpose-e', 'enablement', 'enablement-e', 'purpose-goal'],
    'Evaluation': ['evaluation', 'evaluation-n', 'evaluation-s-e', 'evaluation-s', 'Evaluation',
                   'interpretation', 'interpretation-n', 'interpretation-s-e', 'interpretation-s', 'Interpretation',
                   'conclusion', 'Conclusion', 'comment', 'comment-e', 'evaluation-comment'],
    'Explanation': ['evidence', 'evidence-e', 'explanation-argumentative', 'explanation-argumentative-e', 'Reason', 'reason',
                    'reason-e', "explanation-evidence", "explanation-justify", 'explanation-motivation'],  # justify & motivation: GUM relations
    'Joint': ['List', 'Disjunction', 'joint-disjunction', 'joint-list'],
    'Manner-Means': ['manner', 'manner-e', 'means', 'means-e', 'mode-manner', 'mode-means'],
    'Topic-Comment': ['problem-solution', 'problem-solution-n', 'problem-solution-s', 'Problem-Solution',
                      'Question-Answer', 'question-answer', 'question-answer-n', 'question-answer-s',
                      'Statement-Response', 'statement-response-n', 'statement-response-s',
                      'Topic-Comment', 'Comment-Topic', 'rhetorical-question', "topic-question", "topic-solutionhood", 'organization-phatic'],  # question & solutionhood: GUM relations
    'Summary': ['summary', 'summary-n', 'summary-s', 'restatement', 'restatement-e', 'restatement-partial', 'restatement-repetition'],
    'Temporal': ['temporal-before', 'temporal-before-e', 'temporal-after', 'temporal-after-e', 'Temporal-Same-Time', 'temporal-same-time',
                 'temporal-same-time-e', 'Sequence', 'Inverted-Sequence', 'joint-sequence'],  # lowercase sequence: GUM
    'Topic-Change': ['topic-shift', 'topic-drift', 'Topic-Shift', 'Topic-Drift', "joint-other"],  # joint: GUM relation
    'Textual-Organization': ['TextualOrganization', "organization-heading", 'organization-preparation'],  # preparation: GUM relation
    'span': ['span'],
    'Same-Unit': ['Same-Unit', 'same-unit']  # same-unit - lowercase for GUM
}

relation_map = {None: 0, 'Elaboration': 1, 'Same-Unit': 2, 'Attribution': 3, 'Contrast': 4, 'Cause': 5, 'Enablement': 6,
                'Explanation': 7, 'Evaluation': 8, 'Temporal': 9, 'Manner-Means': 10, 'Joint': 11, 'Topic-Change': 12,
                'Background': 13, 'Condition': 14, 'Comparison': 15, 'Summary': 16, 'Topic-Comment': 17,
                'Textual-Organization': 18}

action_map = {('Shift', None): 0, ('Reduce', 'NS'): 1, ('Reduce', 'NN'): 2, ('Reduce', 'SN'): 3}

xidx_action_map, xidx_relation_map = reverse_dict(action_map), reverse_dict(relation_map)

rel2class = {}
for cl, rels in class2rel.items():
    rel2class[cl] = cl
    for rel in rels:
        rel2class[rel] = cl

