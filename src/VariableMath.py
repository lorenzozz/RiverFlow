import numpy as np


# from Utils import None


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
    return np.std(x)


def vec_load(x, sep: str, enclosed=True):
    """
    Load vector from categorical variable
    :param x: Target categorical variable
    :param sep: Separator of vector string
    :param enclosed: Whether marks like '[' and ']' are present
    :return: the loaded vector
    """
    d = []
    for s in x:
        d.append(np.array(np.fromstring(s[1:-1].strip() if enclosed else s, float, sep=sep)))
    return np.array(d)


def vec_zavadskas(x, verbose=False):
    """
    Computes the Zavadskas-Turskis normalization on the
    target vector. Each sample is normalized as follows.

    x_i' = log(x_i) / sum n=0 to n=N of log(x_n)
    :param x: the target variable
    :param verbose: control normalization output to console
    :return: the normalized vector
    """

    logarithmic = np.log(x)
    norm = np.sum(logarithmic)

    logarithmic = logarithmic / norm
    if verbose:
        print(f"Norm:{norm}")

    return logarithmic


def vec_max_linear(x, verbose=False):
    """ Computes the maximum linear normalization of x.
    The vector is normalized as follows:
        x_i' = (x_i) / max(X)

        :param x: the target vector
        :param verbose: control output to console (print divisor)
        :return: the normalized vector
    """
    stretched = x.reshape(1, np.size(x))[0]
    lin_max = np.max(stretched)

    ret = x / lin_max
    if verbose:
        print(f"Max:{lin_max}")
    return ret


def vec_zscore(x, verbose=False):
    """ Performs z-score normalization on the target vector. The vector is normalized
    as follows:

    x_i = (x_i' - mean(X)) / std(X)

    :param x: the target vector
    :param verbose: control output to console
    :return: the normalized vector
    """
    stretched = x.reshape(1, np.size(x))[0]
    std = np.std(stretched)
    mean = np.mean(stretched)

    ret = (x - mean) / std
    if verbose:
        print(f"Std:{std}, Mean{mean}")
    return ret


def vec_profile(x: np.ndarray):
    """
    Outputs various info about the requested data.
    :param x: The target vector
    :return: No explicit return, output to console
    """
    rsp = x.reshape((1, np.size(x)))[0]
    print(">Count:", np.size(rsp))
    print(">Uniq:", len(set(list(rsp))))
    print(">Range:?")
    print(">Mean:", np.mean(rsp))
    print(">Min:", np.min(rsp))
    print(">Max:", np.max(rsp))
    print(">Median:", np.median(rsp))
    print(">Std:", np.std(rsp))

    return 0


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


def vec_one_hot(x, ordering):
    """
     Returns a vector encoding the target categorical variable x
    according to the ordering passed as parameter
    :param ordering: the ordering of the encoding
    :param x: the target vector
    :returns: one hot encoding
    """
    ordering = np.array(ordering).astype(str)
    o_hot = np.zeros((len(ordering), np.size(x)))
    for label, target in zip(ordering, o_hot):
        target[x == label] = 1.0
    return o_hot.T


def vec_stack(*args):
    """ Returns a pile of stacked vector compatible with the
    windowing algorithm adopted in models"""
    return np.stack([args], axis=1)


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
