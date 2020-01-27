# https://pastebin.com/QKMhafRq
import copy
import random
from functools import partial
from operator import attrgetter

import numpy as np
from deap import gp, tools
from deap.gp import genGrow, genFull, Terminal

from gptsne import tsnedata


def maxheight(v):
    return max(i.height for i in v)


#    return max(attrgetter("height")(itemgetter(0)(v)), attrgetter("height")(itemgetter(1)(v)))
# return itemgetter(0)(attrgetter("height")(v))


# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > tsnedata.max_height:
            new_inds[i] = random.choice(keep_inds)
    return new_inds


def xmate(ind1, ind2):
    # if (random.random() < cxpb):
    i1 = random.randrange(len(ind1))
    i2 = random.randrange(len(ind2))
    ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    return ind1, ind2


def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)


def xmate_st(ind1, ind2):
    # if (random.random() < cxpb):
    i1 = random.randrange(min(len(ind1), len(ind2)))
    # deepcopy needed? duplicates?
    ind1[i1], ind2[i1] = gp.cxOnePoint(copy.deepcopy(ind1[i1]), copy.deepcopy(ind2[i1]))
    return ind1, ind2


def lim_xmate_st(ind1, ind2):
    return wrap(xmate_st, ind1, ind2)


def xmate_addtrees(max_no, ind1, ind2):
    ind1_size = len(ind1)
    i1 = random.randrange(ind1_size)
    ind2_size = len(ind2)
    i2 = random.randrange(ind2_size)

    if ind1_size < max_no:
        ind1.append(copy.deepcopy(ind2[i2]))
    if ind2_size < max_no:
        ind2.append(copy.deepcopy(ind1[i1]))

    return ind1, ind2


def lim_xmate_aic(ind1, ind2):
    """
    Basically, keep only changes that obey max depth constraint on a tree-wise (NOT individual-wise) level.
    :param ind1:
    :param ind2:
    :return:
    """
    keep_inds = [copy.deepcopy(ind1), copy.deepcopy(ind2)]
    new_inds = list(xmate_aic(ind1, ind2))
    for i, ind in enumerate(new_inds):
        for j, tree in enumerate(ind):
            if tree.height > rundata.max_height:
                new_inds[i][j] = keep_inds[i][j]
    return new_inds


def xmate_aic(ind1, ind2):
    min_size = min(len(ind1), len(ind2))
    for i in range(min_size):
        ind1[i], ind2[i] = gp.cxOnePoint(copy.deepcopy(ind1[i]), copy.deepcopy(ind2[i]))
    return ind1, ind2


def xmate_maxt(ind1, ind2):
    max_size = max(len(ind1), len(ind2))
    i1 = random.randrange(max_size)
    i2 = random.randrange(max_size)

    if i1 >= len(ind1):
        # add one!
        ind1.append(copy.deepcopy(ind2[i2]))
    elif i2 >= len(ind2):
        # add one!
        ind2.append(copy.deepcopy(ind1[i1]))
    else:
        # normal crossover.
        ind1[i1], ind2[i2] = gp.cxOnePoint(copy.deepcopy(ind1[i1]), copy.deepcopy(ind2[i2]))

    return ind1, ind2


def lim_xmate_maxt(ind1, ind2):
    return wrap(xmate_maxt, ind1, ind2)


def xmate_bt(ind1, ind2):
    # hmmm
    # if (random.random() < cxpb):
    for i in range(min(len(ind1), len(ind2))):
        ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    #    ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    # if (random.random() < cxpb):
    #   ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])

    return ind1, ind2


def lim_xmate_bt(ind1, ind2):
    return wrap(xmate_bt, ind1, ind2)


def xmate_half_half(max_no, ind1, ind2):
    if random.random() >= 0.5:
        return lim_xmate(ind1, ind2)
    else:
        return xmate_addtrees(max_no, ind1, ind2)


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    ind[i1] = indx[0]
    return ind,


def lim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr)
    # print(res)
    return res


DO_ALL_ERCS = True


def mutate_add_remove(max_no, create_tree, ind):
    curr_num = len(ind)

    # if we can only remove, OR coin flip tells us to remove and we CAN
    if curr_num == max_no or (curr_num > 1 and random.random() >= 0.5):
        #only makes sense to remove the final tree --- as ordering is important.
        del ind[-1]
    else:
        tree = create_tree()
        # print(tree)
        ind.append(tree)
    return ind,


def mutate_erc(ind):
    # See deap.gp.mutEphemeral for inspiration.
    ephemerals_indxs = collect_ephemeral_indices(ind)

    if len(ephemerals_indxs) > 0:
        if DO_ALL_ERCS:
            for i in ephemerals_indxs:
                ind[i[0]][i[1]].value += np.random.normal(0, 0.15)
        else:
            ephemerals_idx = random.choice(ephemerals_indxs)
            chosen = ind[ephemerals_idx[0]][ephemerals_idx[1]]
            # print(chosen.value)
            chosen.value += np.random.normal(0, 0.15)
        # print("A" + str(chosen.value))
    return ind,


def collect_ephemeral_indices(ind):
    ephemerals_indxs = []

    for i in ind:
        ephemerals_idx_i = [(i, index)
                            for index, node in enumerate(ind[i])
                            if (isinstance(node, Terminal) and isinstance(node.value, float))]
        ephemerals_indxs = ephemerals_indxs + ephemerals_idx_i

    return ephemerals_indxs


# @jit(parallel=True)
def str_ind(ind):
    return tuple(str(i) for i in ind)
    # str1 = str(ind[0])
    # str2 = str(ind[1])
    # return str1, str2


def quick_str_tree(tree):
    # string = ""
    # #stack = []
    # for node in tree:
    #     #stack.append((node, []))
    #     while len(stack[-1][1]) == stack[-1][0].arity:
    #         prim, args = stack.pop()
    #         string = prim.format(*args)
    #         if len(stack) == 0:
    #             break  # If stack is empty, all nodes should have been seen
    #         stack[-1][1].append(string)
    #
    # return string

    return str(tree)


#
# def hash_ind_2(ind):
#     return hash(tuple(ind[0]) + tuple(ind[1]))


def hash_ind(ind):
    return hash(str_ind(ind))


# Direct copy from tools - modified for individuals with GP trees in an array
def xselDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first):
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            lind1 = sum([len(gpt) for gpt in ind1])
            lind2 = sum([len(gpt) for gpt in ind2])
            if lind1 > lind2:
                ind1, ind2 = ind2, ind1
            elif lind1 == lind2:
                # random selection in case of a tie
                prob = 0.5

            # Since size1 <= size2 then ind1 is selected
            # with a probability prob
            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter("fitness")))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)


# def xselTournament(individuals, k, tournsize, fit_attr="fitness"):
#     """Select the best individual among *tournsize* randomly chosen
#     individuals, *k* times. The list returned contains
#     references to the input *individuals*.
#
#     :param individuals: A list of individuals to select from.
#     :param k: The number of individuals to select.
#     :param tournsize: The number of individuals participating in each tournament.
#     :param fit_attr: The attribute of individuals to use as selection criterion
#     :returns: A list of selected individuals.
#
#     This function uses the :func:`~random.choice` function from the python base
#     :mod:`random` module.
#     """
#     chosen = []
#     for i in range(k):
#         aspirants = selRandom(individuals, tournsize)
#         chosen.append(max(aspirants, key=attrgetter(fit_attr)))
#     return chosen
def genMTHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_), method(pset, min_, max_, type_)


def lim_genMTHalfAndHalf(pset, min_, max_, type_=None):
    return wrap(genMTHalfAndHalf, pset, min_, max_, type_)
