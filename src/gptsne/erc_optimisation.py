import argparse
import itertools
import math
import operator
import random
import re
from collections import deque
from copy import deepcopy

import cachetools
import numpy as np
from deap import creator, base, cma, tools, algorithms, gp
from deap.gp import Primitive, Terminal, PrimitiveTree
from sklearn.manifold import t_sne
from sklearn.manifold.t_sne import MACHINE_EPSILON

from gptsne import tsnedata
from gptools.ParallelToolbox import ParallelToolbox
from gptsne.eval import evalTSNEMO
from gptools.gp_util import output_ind
from gptsne.deap_stuff import get_pset_weights
from gptools.multitree import collect_ephemeral_indices, str_ind
from gptools.read_data import read_data
from gptsne.tsnedata import perplexity, num_trees


def eval_cmaes(gp_ind, ephemerals_indxs, toolbox, data_t, ind):
    # ind is a list here.
    update_ercs(ephemerals_indxs, gp_ind, ind)

    return evalTSNEMO(data_t, toolbox, gp_ind)


def eval_pso(gp_ind, ephemerals_indxs, toolbox, data_t, reference, ind):
    update_ercs(ephemerals_indxs, gp_ind, ind, reference)
    return evalTSNEMO(data_t, toolbox, gp_ind)


def update_ercs(ephemerals_indxs, gp_ind, ind, reference=None):
    if reference:
        ercs = list(np.array(reference) + np.array(ind))
    else:
        ercs = ind
    for index, val in enumerate(ercs):
        ep_indx = ephemerals_indxs[index]
        node = gp_ind[ep_indx[0]][ep_indx[1]]
        node.value = val
        node.name = str(val)
    str1, str2 = str_ind(gp_ind)
    # print(str1,str2)
    gp_ind.str = str1, str2


def do_cmaes(gp_ind, toolbox, data_t):
    ephemerals_indxs = collect_ephemeral_indices(gp_ind)  # [(0,5),....(1,4),...]
    ephemerals_vals = [gp_ind[indx[0]][indx[1]].value for indx in ephemerals_indxs]
    if not ephemerals_vals:
        return None
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    c_toolbox = base.Toolbox()
    c_toolbox.register("evaluate", eval_cmaes, gp_ind, ephemerals_indxs, toolbox, data_t)
    strategy = cma.Strategy(ephemerals_vals, sigma=0.1)
    c_toolbox.register("generate", strategy.generate, creator.Individual)
    c_toolbox.register("update", strategy.update)

    hof_2 = tools.HallOfFame(1)
    stats_2 = tools.Statistics(lambda ind: ind.fitness.values)
    stats_2.register("avg", np.mean)
    stats_2.register("std", np.std)
    stats_2.register("min", np.min)
    stats_2.register("max", np.max)

    algorithms.eaGenerateUpdate(c_toolbox, ngen=250, stats=stats_2, halloffame=hof_2)
    return hof_2[0]


def do_pso(gp_ind, toolbox, data_t, gens):
    ephemerals_indxs = collect_ephemeral_indices(gp_ind)  # [(0,5),....(1,4),...]
    reference = [deepcopy(gp_ind[indx[0]][indx[1]].value) for indx in ephemerals_indxs]
    if not reference:
        return None, None
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
                   smin=None, smax=None, best=None)
    c_toolbox = base.Toolbox()
    len_ = len(ephemerals_indxs)
    c_toolbox.register("particle", generate, size=len_, pmin=-0.15, pmax=0.15, smin=-0.05, smax=0.05,
                       starting_point=list(itertools.repeat(0., len_)))
    c_toolbox.register("population", tools.initRepeat, list, c_toolbox.particle)
    c_toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)

    c_toolbox.register("evaluate", eval_pso, gp_ind, ephemerals_indxs, toolbox, data_t, reference)

    pop = c_toolbox.population(n=30)
    # make one point the GP best
    pop[0][:] = itertools.repeat(0., len(pop[0]))  # ephemerals_vals
    stats_2 = tools.Statistics(lambda ind: ind.fitness.values)
    stats_2.register("avg", np.mean)
    stats_2.register("std", np.std)
    stats_2.register("min", np.min)
    stats_2.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats_2.fields

    best = None

    for g in range(gens):
        best = eval_and_update(best, c_toolbox, pop)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats_2.compile(pop))
        print(logbook.stream)

    return best, reference


def eval_and_update(best, c_toolbox, pop):
    for part in pop:
        #      print(gp_ind.fitness.values)
        part.fitness.values = c_toolbox.evaluate(part)
        #     print(part.fitness.values)
        if not part.best or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if not best or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values
    for part in pop:
        c_toolbox.update(part, best)
    return best


def generate(size, pmin, pmax, smin, smax, starting_point):
    # some (small) variance from initial position
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part[:] = list(map(operator.add, part, starting_point))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


def from_string_np_terms(string, pset):
    """Try to convert a string expression into a PrimitiveTree given a
    PrimitiveSet *pset*. The primitive set needs to contain every primitive
    present in the expression.

    :param string: String representation of a Python expression.
    :param pset: Primitive set from which primitives are selected.
    :returns: PrimitiveTree populated with the deserialized primitives.
    """
    tokens = re.split("[ \t\n\r\f\v(),]", string)
    expr = []
    ret_types = deque()
    for token in tokens:
        if token == '':
            continue
        if len(ret_types) != 0:
            type_ = ret_types.popleft()
        else:
            type_ = None

        if token in pset.mapping:
            primitive = pset.mapping[token]

            if type_ is not None and not issubclass(primitive.ret, type_):
                raise TypeError("Primitive {} return type {} does not "
                                "match the expected one: {}."
                                .format(primitive, primitive.ret, type_))

            expr.append(primitive)
            if isinstance(primitive, Primitive):
                ret_types.extendleft(reversed(primitive.args))
        else:
            try:
                token = eval(token)
            except NameError:
                raise TypeError("Unable to evaluate terminal: {}.".format(token))

            if type_ is None:
                type_ = type(token)

            _act_type = type(token)
            # THE CHANGE. Can cast float to ndarray.
            if not issubclass(_act_type, type_) and not (_act_type == float and issubclass(type_, np.ndarray)):
                raise TypeError("Terminal {} type {} does not "
                                "match the expected one: {}."
                                .format(token, type(token), type_))
            terminal = Terminal(token, False, type_)
            expr.append(terminal)

            # expr.append(class_)
    return PrimitiveTree(expr)


test_file = '/local/scratch/lensenandr/gpTSNE-Py-MOEAD-V4/ercsAndZeros/dermatologygptsne.gptsne_mo/12/' \
            'dermatology-0.5262551452413216-12792.0.tree'  # 'dermatology-1.0450954770540322-19.0.tree'

test_file = '/local/scratch/lensenandr/ercsZerosNeighboursMeanFixCom2/dermatologygptsne.gptsne_mo/24/' \
            'dermatology-0.488786039491553-2458.0.tree'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset file name", type=str, default="dermatology")
    parser.add_argument("--dir", help="Dataset directory", type=str,
                        default="/home/lensenandr/datasetsPy/")
    parser.add_argument("-g", "--gens", help="Number of generations", type=int, default=1000)
    parser.add_argument("-od", "--outdir", help="Output directory", type=str, default="./")
    parser.add_argument("-f", "--file", help="GP individual file to optimise ERCs of", type=str, default=test_file)
    args = parser.parse_args()
    with open(args.file) as f:
        tree_1_str = f.readline()
        tree_2_str = f.readline()

    all_data = read_data("{}{}.data".format(args.dir, args.dataset))
    data = all_data["data"]
    tsnedata.data_t = data.T
    tsnedata.labels = all_data["labels"]
    num_instances = data.shape[0]
    num_features = data.shape[1]

    pset, weights = get_pset_weights(data, num_features, True, True, True, True)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * tsnedata.nobj)
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)
    toolbox = ParallelToolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalTSNEMO, tsnedata.data_t, toolbox)
    tree1 = from_string_np_terms(tree_1_str, pset)
    tree2 = from_string_np_terms(tree_2_str, pset)
    print(str(tree1))
    print(str(tree2))
    ind = creator.Individual([tree1, tree2])
    print(ind)

    tsnedata.fitnessCache = cachetools.LRUCache(maxsize=1e6)
    tsnedata.outdir = args.outdir
    tsnedata.dataset = args.dataset
    tsnedata.degrees_of_freedom = max(num_trees - 1, 1)
    tsnedata._DOF = (tsnedata.degrees_of_freedom + 1.0) / -2.0
    dists = t_sne.pairwise_distances(data, metric="euclidean", squared=True)
    tsnedata.P_tsne = t_sne._joint_probabilities(dists, perplexity, verbose=True)
    tsnedata.max_P_tsne = np.maximum(tsnedata.P_tsne, MACHINE_EPSILON)

    best, reference = do_pso(ind, toolbox, tsnedata.data_t, args.gens)
    print(best)
    # We still want to output ones that have no constants (even though the vals didn't change!)
    if best is not None:
        ephemerals_indxs = collect_ephemeral_indices(ind)  # [(0,5),....(1,4),...]
        update_ercs(ephemerals_indxs, ind, best, reference)
        ind.fitness.setValues(evalTSNEMO(tsnedata.data_t, toolbox, ind))
    output_ind(ind, toolbox, tsnedata, suffix="-pso", compress=False)
