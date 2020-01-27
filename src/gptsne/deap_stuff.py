import itertools

import math
import numpy as np
from deap import gp
from scipy.spatial import distance_matrix
from sklearn.decomposition import pca

from gptools.gp_util import np_many_add, np_protectedDiv, np_sigmoid, np_relu, np_if, erc_array
from gptsne import tsnedata
from gptsne.tsnedata import NUM_NEIGHBOURS
from gptools.weighted_generators import ProxyArray, RealArray, ZeroArray


def get_pset_weights(data, num_features, use_ercs, use_zeros, use_neighbours, use_neighbours_mean):
    num_var_pca = round(math.sqrt(num_features))
    print("PCA vars: " + str(num_var_pca))
    dat_pca = pca.PCA(copy=True, n_components=1)
    dat_pca.fit(data)
    print(dat_pca)
    pc = dat_pca.components_[0]
    # care about magnitude, not direction
    pc = np.abs(pc)
    ranked_pca_features = np.argsort(-pc)  # sorts highest to smallest magnitude
    print(ranked_pca_features)

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(RealArray, num_features), ProxyArray, "f")

    pset.addPrimitive(np_many_add, [ProxyArray, ProxyArray, ProxyArray, ProxyArray, ProxyArray], RealArray,
                      name="vadd")
    pset.addPrimitive(np.subtract, [ProxyArray, ProxyArray], RealArray, name="vsub")
    pset.addPrimitive(np.multiply, [RealArray, RealArray], RealArray, name="vmul")
    pset.addPrimitive(np_protectedDiv, [RealArray, RealArray], RealArray, name="vdiv")
    pset.addPrimitive(np_sigmoid, [RealArray], RealArray, name="sigmoid")
    pset.addPrimitive(np_relu, [RealArray], RealArray, name="relu")
    pset.addPrimitive(np.maximum, [RealArray, RealArray], RealArray, name="max")
    pset.addPrimitive(np.minimum, [RealArray, RealArray], RealArray, name="min")
    pset.addPrimitive(np_if, [RealArray, RealArray, RealArray], RealArray, name="np_if")
    # deap you muppet
    pset.context["array"] = np.array
    num_ercs = math.ceil(num_features / 10)
    # so we get as many as we do terms...
    if use_ercs:
        print("Using {:d} ERCS".format(num_ercs))
        for i in range(num_ercs):  # range(num_features):
            pset.addEphemeralConstant("rand", erc_array, RealArray)
    weights = {ProxyArray: [], RealArray: []}
    for t in pset.terminals[ProxyArray]:
        weights[ProxyArray].append(t)
        weights[RealArray].append(t)
    if use_zeros:
        print("Using {:d} Zeros".format(num_ercs))
        pset.addTerminal(0, ZeroArray, name="0")
        zero = pset.mapping['0']
        # same number as #of ERCS?
        for i in range(num_ercs):
            weights[ProxyArray].append(zero)
    print("Using top-{:d} features from PCA FS, each {:d} times".format(num_var_pca, num_var_pca))
    for i in range(num_var_pca):
        f_index = ranked_pca_features[i]
        print("Adding feature {:d} from PCA".format(f_index))
        dat_feat = pset.mapping['f' + str(f_index)]
        for j in range(num_var_pca):
            weights[ProxyArray].append(dat_feat)
            weights[RealArray].append(dat_feat)

    if use_neighbours:
        dm = distance_matrix(data, data)
        tsnedata.neighbours = dm.argsort()[:, 1:(1 + NUM_NEIGHBOURS)]
        if use_neighbours_mean:
            for j in range(num_features):
                feat_vals = getNeighbourFeats(0, j,data)
                for i in range(1, NUM_NEIGHBOURS):
                    feat_vals = feat_vals + getNeighbourFeats(i, j,data)
                feat_vals = np.true_divide(feat_vals, NUM_NEIGHBOURS)
                print(feat_vals)
                name = 'nf{}'.format(j)
                print('Adding ' + name)
                pset.addTerminal(np.copy(feat_vals), RealArray, name=name)
                weights[ProxyArray].append(pset.mapping[name])
                weights[RealArray].append(pset.mapping[name])

        else:
            for i in range(NUM_NEIGHBOURS):
                for j in range(num_features):
                    name = 'n{}f{}'.format(i, j)
                    print('Adding ' + name)
                    pset.addTerminal(np.copy(getNeighbourFeats(i, j,data)), RealArray, name=name)
                    weights[ProxyArray].append(pset.mapping[name])
                    weights[RealArray].append(pset.mapping[name])
    # don't forget weights
    return pset, weights


def getNeighbourFeats(n_index, f_index,data):
    these_neighbours = tsnedata.neighbours[:, n_index]
    return data[these_neighbours, f_index]