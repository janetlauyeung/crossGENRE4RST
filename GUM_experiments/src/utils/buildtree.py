from utils.other import rel2class, relation_map


def getparse(tree, parse):
    """ Get parse tree

    :type tree: SpanNode instance
    :param tree: an binary RST tree

    :type parse: string
    :param parse: parse tree in string format
    """
    if (tree.lnode is None) and (tree.rnode is None):
        # Leaf node
        parse += " ( EDU " + str(tree.nuc_edu)
    else:
        parse += " ( " + tree.form
        # get the relation from its satellite node
        if tree.form == 'NN':
            parse += "-" + extract_relation(tree.rnode.relation)
        elif tree.form == 'NS':
            parse += "-" + extract_relation(tree.rnode.relation)
        elif tree.form == 'SN':
            parse += "-" + extract_relation(tree.lnode.relation)
        else:
            raise ValueError("Unrecognized N-S form")
    # print(tree.relation)
    if tree.lnode is not None:
        parse = getparse(tree.lnode, parse)
    if tree.rnode is not None:
        parse = getparse(tree.rnode, parse)
    parse += " ) "
    return parse


def extract_relation(s, level=0):
    """ Extract discourse relation on different level
    """
    # This is to ensure that all the GUM relations are kept intact (the damage is bidirectional)
    if s.lower() in relation_map:  # GUM relations: use GUM-relation sets when available;
        return s.lower()   # lower() fixed for RST-DT multinuclear uppercase
    else:                  # otherwise, use the RST-DT 16 relation classes via rel2class
        return rel2class[s]
