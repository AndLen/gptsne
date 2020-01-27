import glob
import gzip as gz
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special._ufuncs import expit

from gptools.multitree import str_ind


def protectedDiv(left, right):
    if right == 0:
        return 1
    return left / right


def np_protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def np_sigmoid(gamma):
    return expit(gamma)


def np_many_add(a, b, c, d, e):
    return a + b + c + d + e


# https://stackoverflow.com/questions/36268077/overflow-math-range-error-for-log-or-exp
def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    else:
        return 1 / (1 + math.exp(-gamma))


def np_relu(x):
    return x * (x > 0)


def relu(x):
    # fast? https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    return x * (x > 0)


def _if(a, b, c):
    return b if a < 0 else c


def np_if(a, b, c):
    return np.where(a < 0, b, c)


# np...??


def erc_array():
    return random.uniform(-1, 1)


dat_set = set()


def add_to_string_cache(ind):
    hash = str_ind(ind)
    dat_set.add(hash)
    ind.str = hash


def check_uniqueness(ind1, ind2, num_to_produce, offspring):
    # deals with the case where we needed to create two individuals, but had 4 from TS, and the first two were okay,
    if len(offspring) != num_to_produce:
        hash1 = str_ind(ind1)
        if hash1 not in dat_set:
            dat_set.add(hash1)
            ind1.str = hash1
            offspring.append(ind1)
    # the case where we only needed to create 1 more individual, not 2!
    if len(offspring) != num_to_produce:
        hash2 = str_ind(ind2)
        if hash2 not in dat_set:
            dat_set.add(hash2)
            ind2.str = hash2
            offspring.append(ind2)


def output_ind(ind, toolbox, data, suffix="", compress=False, csv_file=None, tree_file=None, del_old=False):
    """ Does some stuff

    :param ind: the GP Individual. Assumed two-objective
    :param toolbox: To evaluate the tree
    :param data: dict-like object containing data_t (feature-major array), outdir (string-like),
    dataset (name, string-like), labels (1-n array of class labels)
    :param suffix: to go after the ".csv/tree"
    :param compress: boolean, compress outputs or not
    :param csv_file: optional path/buf to output csv to
    :param tree_file: optional path/buf to output tree to
    :param del_old: delete previous generations or not
    """
    old_files = glob.glob(data.outdir + "*.tree" + ('.gz' if compress else ''))
    old_files += glob.glob(data.outdir + "*.csv" + ('.gz' if compress else ''))
    out = evaluateTrees(data.data_t, toolbox, ind)
    columns = ['C' + str(i) for i in range(out.shape[1])]
    df = pd.DataFrame(out, columns=columns)
    df["class"] = data.labels

    compression = "gzip" if compress else None

    f_name = ('{}' + ('-{}' * len(ind.fitness.values)) + '{}').format(data.dataset, *ind.fitness.values, suffix)

    if csv_file:
        df.to_csv(csv_file, index=None)
    else:
        outfile = f_name + '.csv'
        if compress:
            outfile = outfile + '.gz'
        p = Path(data.outdir, outfile)
        df.to_csv(p, index=None, compression=compression)

    outfile = f_name + '-aug.csv'
    combined_array = np.concatenate((out, data.data), axis=1)
    aug_columns = columns + ['X' + str(i) for i in range(data.data.shape[1])]
    df_aug = pd.DataFrame(combined_array, columns=aug_columns)
    df_aug["class"] = data.labels
    if compress:
        outfile = outfile + '.gz'
    p = Path(data.outdir, outfile)
    df_aug.to_csv(p, index=None, compression=compression)

    if tree_file:
        tree_file.write(str(ind[0]))
        for i in range(1, len(ind)):
            tree_file.write('\n')
            tree_file.write(str(ind[i]))
    else:
        outfile = f_name + '.tree'
        if compress:
            outfile = outfile + '.gz'

        p = Path(data.outdir, outfile)
        with gz.open(p, 'wt') if compress else open(p, 'wt') as file:
            file.write(str(ind[0]))
            for i in range(1, len(ind)):
                file.write('\n')
                file.write(str(ind[i]))

    if del_old:
        for f in old_files:
            try:
                os.remove(f)
            except OSError as e:  ## if failed, report it back to the user ##
                print("Error: %s - %s." % (e.filename, e.strerror))


def evaluateTrees(data_t, toolbox, individual):
    num_instances = data_t.shape[1]
    num_trees = len(individual)

    if not individual.str or len(individual.str) == 0:
        raise ValueError(individual)

    result = np.zeros(shape=(num_trees, num_instances))

    for i, f in enumerate(individual.str):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=f)
        comp = func(*data_t)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)

        result[i] = comp

    dat_array = result.T
    return dat_array
