from enum import Enum


class RandomDist(Enum):
    """ Map from virtual distinction between distribution to logical different
    functions for computing """
    Normal = 0,
    Logistic = 1,
    Poisson = 2,
    Uniform = 3


def vec_bool_to_num():
    NotImplemented


def vec_inter_outlier():
    NotImplemented


def vec_zero_outliers():
    NotImplemented


def vec_normalize():
    NotImplemented


def vec_mean():
    NotImplemented


def vec_std():
    NotImplemented


def vec_add_noise():
    NotImplemented


def vec_zero_with_prob():
    NotImplemented


def vec_discrete():
    NotImplemented


