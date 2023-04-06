from enum import Enum
import numpy as np


class RandomDist(Enum):
    """ Map from virtual distinction between distribution to logical different
    functions for computing """
    Normal = 0,
    Logistic = 1,
    Poisson = 2,
    Uniform = 3


def vec_bool_to_num(x, dic):

    return np.array([dic[str_val] for str_val in x])


def vec_inter_outlier():
    NotImplemented


def vec_zero_outliers():
    NotImplemented


def vec_zero_mean(x):
    return x - np.mean(x)


def vec_mean(x):
    return np.mean(x)


def vec_std(x):
    return np.mean(x)


def vec_add_noise():
    NotImplemented


def vec_zero_with_prob():
    NotImplemented


def vec_discrete(x, bins, val_map):
    """
    Takes a vector and discretizes its values according to window
    found in bins and mapping values found in val_map
    :param x: the data array
    :param bins:  the bins into which it will partition data
    :param val_map:  the mapping values of each window
    :return:
    """
    result = np.digitize(x, bins)
    val_map.insert(0, None)
    result = np.array([val_map[map_val] for map_val in result])
    return result
