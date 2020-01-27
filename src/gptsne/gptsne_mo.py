import argparse
import time

from deap import base
from deap import creator
from deap.tools import ParetoFront
from sklearn.manifold import t_sne
from sklearn.manifold.t_sne import MACHINE_EPSILON

import gptools.weighted_generators as wg
from gptools.ParallelToolbox import ParallelToolbox
from gptools.gp_util import *
from gptools.multitree import *
from gptools.read_data import read_data
from gptsne import gptsne_moead, tsnedata
from gptsne.deap_stuff import get_pset_weights
from gptsne.erc_optimisation import do_pso
from gptsne.eval import evalTSNEMO
from gptsne.tsnedata import nobj, max_depth, pop_size, cxpb, mutpb, mutercpb, post_cmaes, perplexity, num_trees


def main():
    pop = toolbox.population(n=pop_size)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(lambda ind: ind.fitness.values[1])
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", np.min, axis=0)
    mstats.register("median", np.median, axis=0)
    mstats.register("max", np.max, axis=0)
    # varAnd or varOr
    hof = ParetoFront()
    this_moead = gptsne_moead.GPTSNEMOEAD(tsnedata.data_t, pop, toolbox, len(pop), cxpb, mutpb, tsnedata, ngen=NGEN,
                                          stats=mstats,
                                          halloffame=hof, verbose=True, adapative_mute_ERC=False)
    pop, logbook, hof = this_moead.execute()

    return pop, mstats, hof, logbook


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logfile", help="log file path", type=str, default="log.out")
    parser.add_argument("-d", "--dataset", help="Dataset file name", type=str, default="iris")
    parser.add_argument("--dir", help="Dataset directory", type=str,
                        default="datasets/")
    parser.add_argument("-g", "--gens", help="Number of generations", type=int, default=1000)
    parser.add_argument("-od", "--outdir", help="Output directory", type=str, default="./")
    parser.add_argument("--erc", dest='use_ercs', action='store_true')
    parser.add_argument("--zeros", dest='use_zeros', action='store_true')
    parser.add_argument("--neighbours", dest="use_neighbours", action="store_true")
    parser.add_argument("--neighbours-mean", dest="use_neighbours_mean", action="store_true")
    parser.set_defaults(use_ercs=False)
    parser.set_defaults(use_zeros=False)
    parser.set_defaults(use_neighbours=False)
    parser.set_defaults(use_neighbours_mean=False)
    args = parser.parse_args()
    filename = Path(args.dir).joinpath(args.dataset + '.data')
    all_data = read_data(filename)
    data = all_data["data"]
    num_instances = data.shape[0]
    num_features = data.shape[1]
    tsnedata.labels = all_data["labels"]
    tsnedata.data = data
    tsnedata.data_t = data.T

    pset, weights = get_pset_weights(data, num_features, args.use_ercs, args.use_zeros, args.use_neighbours,
                                     args.use_neighbours_mean)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * nobj)
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox = ParallelToolbox()  #
    toolbox.register("mate", lim_xmate_st)

    toolbox.register("expr", wg.w_genHalfAndHalf, pset=pset, weighted_terms=weights, min_=0, max_=max_depth)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=num_trees)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalTSNEMO, tsnedata.data_t, toolbox)
    toolbox.register("select", tools.selNSGA2)

    toolbox.register("expr_mut", wg.w_genFull, weighted_terms=weights, min_=0, max_=max_depth)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)
    toolbox.register("mutate_erc", mutate_erc)

    tsnedata.outdir = args.outdir
    tsnedata.dataset = args.dataset
    tsnedata.degrees_of_freedom = max(num_trees - 1, 1)
    tsnedata._DOF = (tsnedata.degrees_of_freedom + 1.0) / -2.0
    dists = t_sne.pairwise_distances(data, metric="euclidean", squared=True)
    tsnedata.P_tsne = t_sne._joint_probabilities(dists, perplexity, verbose=True)
    tsnedata.max_P_tsne = np.maximum(tsnedata.P_tsne, MACHINE_EPSILON)

    print(args)

    # Number of generations
    NGEN = args.gens

    assert math.isclose(cxpb + mutpb + mutercpb, 1), "Probabilities of operators should sum to ~1."
    start = time.time()
    pop, stats, hof, logbook = main()
    end = time.time()

    print("TIME: " + str(end - start) + "s")

    for res in hof:
        output_ind(res, toolbox, tsnedata, compress=False)

    p = Path(tsnedata.outdir, args.logfile + '.gz')
    with gz.open(p, 'wt') as file:
        file.write(str(logbook))

    pop_stats = [str(p.fitness) for p in pop]
    pop_stats.sort()
    hof_stats = [str(h.fitness) for h in hof]

    # hof_stats.sort()
    print("POP:")
    print("\n".join(pop_stats))

    print("PF:")
    print("\n".join(hof_stats))


    # do it all after so if it's too slow at least the above is outputted already
    if post_cmaes:
        for res in hof:
            cmaes_res = do_pso(res, toolbox, tsnedata.data_t, 1000)
            if cmaes_res:  # since this is just the vector of consts...
                ephemerals_indxs = collect_ephemeral_indices(res)  # [(0,5),....(1,4),...]
                for index, val in enumerate(cmaes_res):
                    ep_indx = ephemerals_indxs[index]
                    res[ep_indx[0]][ep_indx[1]].value = val
                res.fitness.setValues(evalTSNEMO(tsnedata.data_t, toolbox, res))
            output_ind(res, toolbox, tsnedata, "-pso")
