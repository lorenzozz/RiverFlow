import numpy as np


def vec_bool_to_num(x, dic):
    """
    Maps a categorical/boolean data onto a dictionary
    :param x: target vector
    :param dic: dictionary containing mapping values for each category
    :return: the vector mapped onto the numbers
    """
    return np.array([dic[str_val] for str_val in x])


def vec_inter_outlier(x, m):
    """
    Reference: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    :param x: vector
    :param m: algorithm parameter
    :return: vector with outliers brought to the mean
    """
    a = np.copy(x)
    d = np.abs(a - np.median(a))
    m_dev = np.median(d)
    s = d / m_dev if m_dev else np.zero(len(d))
    a[s > m] = m * m_dev
    return a


def vec_zero_outliers(x, n):
    """
    Zeroes out the elements such that x[i] > std*n

    :param x: the target vector
    :param n: the amount of stds
    :return: the vector with its outliers zeroed out
    """
    a = np.copy(x)
    dev_stand = np.std(x)
    a[a > n * dev_stand] = 0
    return a


def vec_zero_mean(x):
    """
    Subtracts the mean of vector x to x itself, making it a
    zero-mean vector.
    :param x: variable
    :return: x with zero mean
    """
    return x - np.mean(x)


def vec_mean(x):
    """
    Computes the mean of a variable
    :param x: variable
    :return: variable's mean
    """
    return np.mean(x)


def vec_shuffle(x):
    """
    Randomly shuffles the target array
    :param: the target vector
    :return: a shuffled vector
    """
    # Necessary, as np.random.shuffle alters the original array
    a = np.copy(x)
    np.random.shuffle(a)
    return a


def vec_std(x):
    """
    Computes the standard deviation of a variable
    :param x: variable
    :return: variable's standard deviation
    """
    return np.mean(x)


def vec_add_noise(x, distribution, *args):
    """
    Adds noise of the requested distribution, mean and std (varying on distribution)
    to the parameter vector x
    :param x: target vector
    :param distribution: name of distribution
    :param args: situational arguments
    :return: vector with noise added
    """
    noise_x = None
    if distribution == "gaussiana":
        noise_x = x + args[0] + np.random.randn(np.size(x)) * args[1]
    elif distribution == "esponenziale":
        noise_x = x + np.random.exponential(args[0], np.size(x))
    elif distribution == "uniforme":
        noise_x = x + np.random.uniform(args[0], args[1], np.size(x))
    return noise_x


def vec_zero_with_prob(x, probability):
    """
    Zeroes out random elements in p with probability <probability>
    :param x: target vector
    :param probability: probability that a random element x[i] is zeroed
    :return: the result vector
    """
    a = x
    indices = np.random.choice(np.arange(np.size(x)), replace=False,
                               size=int(np.size(x) * probability))
    a[indices] = 0

    return a


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


def vec_one_hot(x):
    a = set(x)
    r = np.stack(axis=-1)
    NotImplemented

def vec_interval(a, b, n):
    """
    Returns n evenly spaced sample from linear interval[a,b]
    :param a: starting value
    :param b: ending value
    :param n: amount of samples
    :return: linear range between [a,b] with n samples
    """
    return np.linspace(a, b, n)


def vec_truncate(x, n):
    a = x[:n]
    return a
