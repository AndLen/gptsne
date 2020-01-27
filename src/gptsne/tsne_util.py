from time import time

import numpy as np
from MulticoreTSNE import MulticoreTSNE
from numba import jit
from scipy.spatial.distance import pdist
from sklearn.manifold import t_sne
from sklearn.manifold.t_sne import MACHINE_EPSILON
from sklearn.neighbors import NearestNeighbors


def pairwise_distance_nn(data, perplexity, verbose=True):
    # Cpmpute the number of nearest neighbors to find.
    # LvdM uses 3 * perplexity as the number of neighbors.
    # In the event that we have very small # of points
    # set the neighbors to n - 1.
    n_samples = data.shape[0]

    k = min(n_samples - 1, int(3. * perplexity + 1))

    if verbose:
        print("[t-SNE] Computing {} nearest neighbors...".format(k))

    # Find the nearest neighbors for every point
    knn = NearestNeighbors(algorithm='auto', n_neighbors=k,
                           metric="euclidean")
    t0 = time()
    knn.fit(data)
    duration = time() - t0
    if verbose:
        print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
            n_samples, duration))

    t0 = time()
    distances_nn, neighbors_nn = knn.kneighbors(
        None, n_neighbors=k)
    duration = time() - t0
    if verbose:
        print("[t-SNE] Computed neighbors for {} samples in {:.3f}s..."
              .format(n_samples, duration))

    # Free the memory used by the ball_tree
    del knn

    # if metric == "euclidean":
    # knn return the euclidean distance but we need it squared
    # to be consistent with the 'exact' method. Note that the
    # the method was derived using the euclidean method as in the
    # input space. Not sure of the implication of using a different
    # metric.
    distances_nn **= 2
    return distances_nn, neighbors_nn


def tsneError_MultiCpore_Tsne(params, data, perplexity):
    # need to not recreate P
    mtSNE = MulticoreTSNE(perplexity=perplexity, init=params, n_iter=0, early_exaggeration=1)
    mtSNE = mtSNE.fit(data)
    error = mtSNE.kl_divergence_
    print(error)
    # Yeah it's a hack.
    return error


def tsneError_bh_full(params, P, degrees_of_freedom):
    X_embedded = params  # params.reshape(n_samples, n_components)
    num_instances = X_embedded.shape[0]
    num_trees = X_embedded.shape[1]
    res = t_sne._kl_divergence_bh(params, P, degrees_of_freedom, num_instances, num_trees)
    return (res[0])


@jit(nopython=True, parallel=True)
def do_dot(a, b):
    return np.dot(a, b)


def tsneError(params, P, degrees_of_freedom, max_P, _DOF):
    X_embedded = params

    # Q is a heavy-tailed distribution: Student's t-distribution
    Q = do_Q(X_embedded, _DOF, degrees_of_freedom)
    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS
    # max_P = np.maximum(P, MACHINE_EPSILON)
    # Objective: C (Kullback-Leibler divergence of P and Q)
    q = do_div(Q, max_P)
    log = do_log(q)
    # for profiling
    dot = do_dot(P, log)

    kl_divergence = 2.0 * dot
    return kl_divergence


# @jit
def do_Q(X_embedded, _DOF, degrees_of_freedom):
    dist = pdist(X_embedded, "sqeuclidean")
    Q = do_Q_2(_DOF, degrees_of_freedom, dist)
    return Q


@jit(nopython=True)
def do_Q_2(_DOF, degrees_of_freedom, dist):
    if degrees_of_freedom != 1:
        dist /= degrees_of_freedom
    dist += 1.
    dist **= _DOF
    sum_dist = dist_sum(dist)
    norm_dist = dist_norm(dist, sum_dist)
    Q = max_Q(norm_dist)
    return Q


@jit(nopython=True, parallel=True)
def max_Q(norm_dist):
    Q = np.maximum(norm_dist, MACHINE_EPSILON)
    return Q


@jit(nopython=True)
def dist_norm(dist, sum_dist):
    norm_dist = dist / sum_dist
    return norm_dist


@jit(nopython=True)
def dist_sum(dist):
    sum_dist = np.sum(dist)
    sum_dist *= 2
    return sum_dist


@jit(nopython=True)
def do_log(q):
    log = np.log(q)
    return log


@jit(nopython=True)
def do_div(Q, max_P):
    q = max_P / Q
    return q
