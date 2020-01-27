import random

# see gp.py
import sys
from inspect import isclass

from deap.creator import _numpy_array


def w_genFull(pset, weighted_terms, min_, max_, type_=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return w_generate(pset, weighted_terms, min_, max_, condition, type_)


def w_genGrow(pset, weighted_terms, min_, max_, type_=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth == height or \
               (depth >= min_ and random.random() < pset.terminalRatio)

    return w_generate(pset, weighted_terms, min_, max_, condition, type_)


def w_genHalfAndHalf(pset, weighted_terms, min_, max_, type_=None):
    #TODO: limit size?
    method = random.choice((w_genGrow, w_genFull))
    return method(pset, weighted_terms, min_, max_, type_)


def w_generate(pset, weighted_terms, min_, max_, condition, type_=None):
    if type_ is None:
        type_ = pset.ret

    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                if type_ is ZeroArray:
                    #anywhere a zero can go, a proxy can also go....don't replace it with just another zero
                    type_ = ProxyArray
                term = random.choice(weighted_terms[type_])
                #term = random.choice(weighted_terms)
                # term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add " \
                                 "a terminal of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                if type_ is ZeroArray:
                # we have no functions that produce zeros...but anything taking a zero can also take a real.
                    type_ = ProxyArray
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add " \
                                 "a primitive of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr

# this makes sure things get copied correctly....
class ProxyArray(_numpy_array):
    pass


class RealArray(ProxyArray):
    pass


class ZeroArray(ProxyArray):
    pass