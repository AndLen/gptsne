from gptools.gp_util import evaluateTrees
from gptools.util import cachedError
from gptools.array_wrapper import ArrayWrapper
from gptools.weighted_generators import ZeroArray
from gptsne import tsnedata
from gptsne.tsne_util import tsneError


def compute_complexity(individual):
    nz_count_0_0 = sum(1 for n in individual[0] if n.ret is ZeroArray)
    nz_count_0_1 = sum(1 for n in individual[1] if n.ret is ZeroArray)
    count_0 = len(individual[0])
    count_1 = len(individual[1])
    return (count_0 + count_1) - (nz_count_0_0 + nz_count_0_1)


def evalTSNEMO(data_t, toolbox, individual):
    dat_array = evaluateTrees(data_t, toolbox, individual)
    hashable = ArrayWrapper(dat_array)
    args = (dat_array, tsnedata.P_tsne,
            tsnedata.degrees_of_freedom, tsnedata.max_P_tsne,
            tsnedata._DOF)

    badness = cachedError(hashable, tsneError, tsnedata,args=args,kargs={})
    to_return = badness, compute_complexity(individual)

    return to_return
