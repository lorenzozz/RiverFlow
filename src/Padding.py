# imports
import copy
import random
import typing
import warnings
import numpy as np
import tensorflow as tf
import Utils

from abc import abstractmethod
from Config import *
from DataOrganizer import DataFormatReader
from matplotlib import pyplot as plt
from typing import Iterable
import locale

DEFAULT_VERTICAL_WINDOW_SIZE = 14
DEFAULT_DENSITY_VALUE = 0.7


class ToNumpyMasked:
    def __init__(self, v, allowed):
        """ Create masked vector. Only a subset defined by allowed
        can be modified. """
        self._x = v
        self._allowed = allowed

    def get_allowed(self):
        """ Get allowed elements from original vector. """
        return np.array(self._x)[self._allowed]

    def mask(self, v):
        return np.array(v)[self._allowed]

    def update(self, quantity):
        """ Update masked vector with quantity of mask size. """
        for i, q in zip(self._allowed, quantity):
            self._x[i] = q

    @property
    def python_rep(self):
        """ Get original representation. """
        return self._x

    @python_rep.setter
    def python_rep(self, v):
        """ Change representation """
        self._x = v


class FormatParser:
    """
    Namespace for both string format parsing routines
    and plan description parsing routines.
    No explicit initialization is required as all methods are
    class methods.
    """

    @classmethod
    def multi_split(cls, string, splits: Iterable):
        c = [string]
        for sep in splits:
            c = [piece for part in c for piece in part.split(sep)]
        return c

    @classmethod
    def get_separators(cls, string: str, splits: Iterable):
        """
        Parse expressions of the type
        x_1{ID1}x_2{ID2}...
        separating the IDS inside the split characters ('{', '}')
        and the textual separators between them.

        :param string: The target expression
        :param splits: The splitting elements
        :return: The textual separators in accordance with
        the splitting characters
        """
        return cls.multi_split(string, splits)[::2]

    @classmethod
    def get_names(cls, string: str, splits: Iterable):
        """
        Parse expressions of the type
        x_1{ID1}x_2{ID2}...
        separating the IDS inside the split characters ('{', '}')
        and the textual separators between them.

        :param string: The target expression
        :param splits: The splitting elements
        :return: The IDs enclosed in the splitting characters
        """
        return cls.multi_split(string, splits)[1::2]


class NoiseGenerator:
    @staticmethod
    def zero_with_p(data_set, p, val=0):
        """ Zero out elements at random. """
        for i in range(0, len(data_set)):
            for j in range(0, len(data_set[0])):
                if (random.randint(1, 100)) / 100 < p:
                    data_set[i][j] = val

    @staticmethod
    def add_noise_samples(dataset, amount):
        """ Sample with replacement and add noise amount times. """
        taken = copy.deepcopy(dataset)
        for _ in range(amount):
            a = random.choice(dataset)
            taken.append(a)
            a = a + np.random.randn(len(dataset[0]))
            dataset.append(a)

        return taken

    @staticmethod
    def add_zero_samples(dataset, amount, val=0):

        taken = copy.deepcopy(dataset)
        for _ in range(amount):
            a = random.choice(dataset)
            taken.append(a)
            for i in range(0, len(a)):
                p = random.randint(0, 10)
                if random.randint(0, 10) > p:
                    a[i] = val
            dataset.append(a)

        return taken


class ImputationModel:
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """ Create new imputation model """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """ Get name of model """
        pass

    @abstractmethod
    def __call__(self, d_point, sentinel) -> typing.Any:
        """ Make imputation on data """
        pass

    @abstractmethod
    def get_error(self, fca, test_set) -> float:
        """ Get model error on source """
        pass


@typing.final
class HorizontalZeroImputationModel(ImputationModel):
    def __str__(self) -> str:
        return "Zero imputation on row."

    def __init__(self, _fca, allowed, _hypers):
        self._local_mask = ToNumpyMasked(None, allowed)

    def __call__(self, data_row, sentinel):
        """ Zero out missing values in a datapoint """
        # Note that numpy casts to string all elements to make
        # an array homogeneous
        data: np.array = self._local_mask.mask(data_row)
        data[data == sentinel] = 0.0
        self._local_mask.python_rep = data_row
        self._local_mask.update(data)
        return self._local_mask.python_rep

    def get_error(self, fca, test_set) -> float:
        return 0


@typing.final
class VerticalZeroImputationModel(ImputationModel):
    def __str__(self) -> str:
        return "Vertical Zero Imputation"

    def __init__(self, _fca, __allowed, ___hypers):
        """ No initialization is required. """
        pass

    def __call__(self, _datapoint, __sentinel):
        """ Return zero for each missing value. """
        return 0.0

    def get_error(self, fca, test_set) -> float:
        return 0


@typing.final
class VerticalWeightedAverageModel(ImputationModel):

    def __str__(self) -> str:
        return "Vertical weighted average"

    def __init__(self, _fca, __allowed, hypers):
        """
        Initialize a vertical weighted average padding model.

        :param _fca: not available
        :param __allowed: same as above
        :param hypers: Hyperparameters such as distance between days
            in the time series.
            Any function provided by the user must go under the label
            'distance' and must have the following signature

            def user_distance(index_1, index_2, x_1, x_2)
                ...
                return float(...)

            If no window is specified (as label 'w'), a default
            value of the vertical window is assumed.
        """

        def _def_distance(i_1, i_2, _x_1, _x_2):
            return np.abs(i_1 - i_2)

        self._distance = _def_distance
        if hypers and 'distance' in hypers:
            self._distance = hypers['distance']
        if not hypers or 'w' not in hypers:
            self._w = DEFAULT_VERTICAL_WINDOW_SIZE
        else:
            self._w = hypers['w']

    def __call__(self, data_row, sentinel):
        """
        Get weighted average vertical imputation.

        :param data_row: The target data row from the dataset
        :param sentinel: The sentinel value for missing features
        :return: The resulting imputed feature
        """
        d = np.array(data_row)
        indexes = np.flatnonzero(np.array(data_row) != sentinel) - self._w
        weights = [1 / (self._distance(0, i, data_row[self._w - 1], data_row[i + self._w]) + 1e-7) for i in indexes]
        norm = np.sum(weights) + 1e-7
        res = np.dot(weights, d[d != sentinel].astype(float)) / norm
        return res

    def get_error(self, fca, test_set) -> float:
        return 0


@typing.final
class HorizontalKNNModel(ImputationModel):

    def __init__(self, fca, allowed, hypers):
        """
        Initialize a horizontal k-nearest-neighbors padding model.

        :param fca: The data to compare against new samples
        :param allowed: The features which it is allowed to alter
        :param hypers: Hyperparameters such as k and eta.
            if no k is provided, a shallow search for a good enough
            k is autonomously executed.
        """

        # Euclidean distance (Minkowski 2)
        def _def_distance(dp_1, dp_2):
            return np.linalg.norm(np.array(dp_1) - np.array(dp_2))

        self._local_mask = ToNumpyMasked(None, allowed)
        self._distance = _def_distance
        if hypers and 'distance' in hypers:
            self._distance = hypers['distance']

        self._pool = np.array(np.array(
            [self._local_mask.mask(dp) for dp in fca]).astype(np.float32))

        if not hypers or 'k' not in hypers:
            self._k = self._get_best_k()
        else:
            self._k = hypers['k']

    def __call__(self, data_row, sentinel):
        """
        Get weighted average vertical imputation.
        The actual prediction is computed with a weighted and
        normalized k-nearest neighbor algorithm. That is, each
        missing row is computed as

        m_hat = 1 / (sum of weights) * sum_k of weights_k*neighbor_k
        where the candidates ore the k nearest neighbors.
        Note that features not missing are not altered by this
        imputation.

        :param data_row: The target data row from the dataset
        :param sentinel: The sentinel value for missing features
        :return: The resulting imputed feature
        """

        def _masked_euclidean(x_1, x_2):
            non_missing_samples = np.nonzero(x_1 != sentinel)
            return np.linalg.norm(x_1[non_missing_samples], x_2[non_missing_samples])

        masked_dp = self._local_mask.mask(data_row)
        prediction = self._run_knn(self._pool, masked_dp, _masked_euclidean)

        for i in range(0, len(masked_dp)):
            if masked_dp is not sentinel:
                prediction[i] = masked_dp[i]

        self._local_mask.python_rep = data_row
        self._local_mask.update(prediction)

        return self._local_mask.python_rep

    def _run_knn(self, fca, datapoint, distance: typing.Callable):
        """
        Runs the KNN algorithm on the complete datapoints in fca
        wrt the datapoint provided as argument, using the distance
        metric passed as argument.

        Returns the actual weighted average, thus it's a workhorse
        function for call() (and _get_best_k)

        :param fca: The dataset of complete values
        :param datapoint: The value to impute
        :param distance: The distance metric employed
        :return: The computed value to impute
        """

        def _get_neighbours():
            dists = [distance(datapoint, sample) for sample in fca]
            epsilon = 1e-5

            def _sort_together(l1, l2):
                # l1 and l2 has to be numpy arrays
                idx = np.argsort(l1).astype(int)
                return l1[idx], l2[idx]

            dists_sort, index_sort = _sort_together(np.array(dists), np.arange(0, len(dists)))
            weights = 1 / (np.array(dists_sort[:self._k]) + epsilon)

            return weights, dists_sort, index_sort

        # Compute weighted and normalized sum
        ws, ds, indexes = _get_neighbours()
        norm = np.sum(ws)
        candidates = np.array(fca)[indexes[:self._k]]
        new_x = candidates[0] * ws[0]
        for i in range(1, self._k):
            new_x += candidates[i] * ws[i]
        new_x /= norm

        return new_x

    def _get_best_k(self):
        """
        Shallow heuristic to compute a good enough value of k.
        For each time step, k is incremented linearly by

            k_t+1 = alpha*k + beta

        If the average error over cross validation batches is
        greater than the average error of the previous value for k,
        a mean value between the previous k and the current k
        is taken according to the equation

            k_hat = k_t + eta * (err_t / err_t_p_1) * (k_t_p_1 - k_t)

        for numerical purposes the denominators are corrected with an
        epsilon factor of 1.e-7
        :return: A good value of k
        """
        k_t = k_t_p_1 = 16
        error_t_p_1 = 0
        eta, alpha, beta = 0.75, 1.5, 0.0
        upper_bound = 120

        # Not enough data to conduct search.
        if len(self._pool) < 100:
            # Set k = floor [ len(dataset) / 10 ]
            return np.floor(len(self._pool) / 10)
        # Else prepare for cross-validation testing

        batches = np.array_split(self._pool, 5)
        error_t = np.inf
        while error_t >= error_t_p_1 and k_t_p_1 < upper_bound:
            error_t = error_t_p_1
            k_t = k_t_p_1

            test_i = random.randint(0, 4)
            test_batch = batches[test_i]
            avg_batches = [batches[i] for i in range(0, 5) if i != test_i]

            k_t_p_1 = np.floor(alpha * k_t + beta)
            self._k = int(k_t_p_1)
            errors = [self.get_error(batch, test_batch) for batch in avg_batches]
            error_t_p_1 = np.sum(errors) / len(avg_batches)

        final_k = np.ceil(k_t + eta * (error_t / error_t_p_1 + 1.e-4) * (k_t_p_1 - k_t))
        print("Final k:", final_k)

        return final_k

    def get_error(self, fca, test_set) -> float:
        """
        Get the MSE of the test set by randomly adding noise
        to it and verifying the degree of accuracy induced by
        imputing with training set given by full cases provided.

        :param fca: The complete training samples.
        :param test_set: The  complete samples to test.
        :return: the MSE of the test.
        """
        n = len(test_set)
        artificial_sent = -1050325

        def _induced_euclidean(x_1, x_2):
            x_1 = np.array(x_1).astype(np.float32)
            x_2 = np.array(x_2).astype(np.float32)
            non_missing_samples = np.flatnonzero(x_1 != artificial_sent)
            return np.linalg.norm(x_1[non_missing_samples] - x_2[non_missing_samples])

        noisy_set = copy.deepcopy(test_set).astype(float)
        NoiseGenerator.zero_with_p(noisy_set, 0.3)
        results = []
        for test_dp in test_set:
            results.append(self._run_knn(fca, test_dp, distance=_induced_euclidean))

        mse = 0.0
        for g_truth, predicted in zip(test_set, noisy_set):
            mse += (np.square((g_truth - predicted))).mean()
        print("MSE: ", mse / n)
        del noisy_set
        return mse / n

    def __str__(self) -> str:
        return "Horizontal KNN model"


class Dataset:
    def __init__(self,
                 init_source: tuple[str, str] = None):
        """
        Creates a new Dataset object. If no initial source is provided,
        an empty dataset is generated instead.

        :param init_source: Optional initial source of data
        """
        self._sources_specs = []
        self._variables = []

        if init_source is not None and not isinstance(init_source, Iterable):
            raise ValueError("Attempting to initialize a Dataset object with "
                             "an incorrect initial source. Both a source path"
                             "and a format string must be provided if a source"
                             "is to be added in the constructor.")

        if init_source is not None:
            path, format_string = init_source
            self.load_source(path, format_string)
        else:
            self._data_sources = []

    def load_plan(self, path: str, unroll_window=False, unroll_specs=None):
        pass

    def load_source(self, path: str, format_string: str, **kwargs):
        """
        Load dataset from source file. The target file must be a textual file.
        No assumption is made on the actual structure of the file. The
        parser can be completely configured in the formast_string to accept
        any format as long as the file is textual and structured row-wise.

        The function has no explicit returns, as it simply alters internal
        representation of data.

        :param path: The target file path
        :param format_string: The format of each row the target file
        """

        f_seps = FormatParser.get_separators(format_string, ['{', '}'])
        self._data_sources.append(DataFormatReader.parse_file(
            path,
            f_seps)
        )

        var_names = FormatParser.get_names(format_string, ['{', '}'])
        self._variables.append(var_names)

        spec = Dataset._check_spec_kwargs(**kwargs)
        self._sources_specs.append(spec)

    @staticmethod
    def _check_spec_kwargs(**kwargs):
        """
        Collect specifications for each source. An unrecognized
        specification is taken as a fatal error. No attempt to recover
        is made.

        Recognized specifications at the moment are:

            - time_series: Indicates whether a source represents a time series.
                A string containing the label of the variable containing
                dates in the source must be provided as a keyword argument.

            - has_missing_values: Indicates whether a source has missing values
                and should be altered by imputation procedures. A value
                representing the missing sentinel must be specified as keyword
                argument along with the key.

            - load_desc: Indicates whether the user requests a specific
                description to be loaded and for it not to be generated
                automatically. The description or a filepath to the description
                must follow after this key.

            - pad_missing-days: Indicates whether the imputation of missing
                values should also add missing days in the time series. A
                value of True must follow this key.

            - date_format: Specifies the format for the date in the time
                series if the source is indicated to be one.

        :param kwargs: The user requested specifications
        :raises: ValueError on unrecognized specs
        :return: The verified specifications
        """

        recognized_specifications = [
            'time_series',
            'has_missing_values',
            'load_desc',
            'pad_missing_days',
            'date_format'
        ]

        for prop in kwargs.keys():
            if prop not in recognized_specifications:
                raise ValueError("Attempting to give a source the specification"
                                 "{spec} which was not recognized. Please see "
                                 "the loader's __docs__ for a complete list of "
                                 "allowed specifications.")

        return kwargs

    @property
    def raw_data(self):
        return self._data_sources

    @raw_data.setter
    def raw_data(self, _):
        warnings.warn("Attempting to manually assign a value to "
                      "a Dataset is illegal. Use load_source or "
                      "load_plan instead.")

    def flush_sources(self):
        """
        Reset Dataset to an empty dataset. Attribute other than
        self._variables and _data_sources are not altered during
         the flush as they can only be set once.
        """
        self._variables = []
        self.delete()

    @staticmethod
    def _pad_missing_days(src_spec, variables, src):
        """
        Workhorse function for make_rectangular. Finds missing
        elements in a time series (day-wise) and pads the series with
        empty vectors containing all missing sentinels for each feature.

        :param src_spec: The specific of the source data
        :param variables: The variables involved
        :param src: The source data vector
        :return: Pads missing days into source.
        """
        feature_amt = len(variables)
        missing_sentinel = src_spec['has_missing_values']
        d_format, d_label = src_spec['date_format'], src_spec['time_series']
        date_i = variables.index(d_label)

        just_dates = [datapoint[date_i] for datapoint in src]
        missing = Utils.check_series_for_missing_days(
            just_dates,
            d_format
        )
        for m in missing:
            index, rep = Utils.find_first_before(
                just_dates,
                d_format,
                m
            )
            # Add vector full of missing sentinels for each feature.
            missing_rep = [missing_sentinel] * feature_amt
            missing_rep[date_i] = rep
            # Insert it into the right place of the array. it is
            # necessary to also update just_dates to get the right indexes
            # to pad even while appending to the original source.
            src.insert(
                index + 1,
                missing_rep
            )
            just_dates.insert(index + 1, rep)

    def make_rectangular(self, pad_alg_specs=None,
                         to_obj: dict[str, typing.Callable] = None,
                         horizontal_padding_methods: str = 'best',
                         vertical_padding_methods: str = 'best',
                         hyper_parameters: dict = None,
                         user_default_cast: typing.Callable = None):
        """
        Make dataset 'rectangular' by padding any missing datapoints.
        Any sub-dataset containing missing datapoints must be tagged
        as such in the source specification for the procedure to
        alter it.

        The available imputation methods available differ from horizontal
        to vertical padding.
        - Horizontal methods: knn, dae, zero
        - Vertical methods: wa, dae, zero

        Vertical imputation methods only make sense for time series data, as
        categorical or simply time-uncorrelated variables gain no benefit for
        having missing data features imputed based on proximity to other
        samples in the dataset.

        It is possible to specify the threshold for determining the choice
        between vertical to horizontal methods.
        Vertical methods are used, when possible on time series, when
        the number of full-case samples in the proximity of the incomplete
        datapoint is bigger or equal than N_x.

        The default method is 'best', meaning all methods are tested and
        a cross-validations testing phase determines the best fit for the
        dataset.
        Optional hyperparameter tuning by hand is possible by passing them
        as keys into the hyper dictionary. Note that most hyperparameters
        are regulated internally autonomously.

        :param user_default_cast: An optional user defined default
            to casting method
        :param pad_alg_specs: The specific of the imputation algorithm
            as in the docs.
        :param to_obj: The routines mapping strings to the objects.
            Defaults to casting to float.
        :param horizontal_padding_methods: The horizontal imputation methods applied.
        :param vertical_padding_methods: The vertical imputation methods applied.
        :param hyper_parameters: The optional hyperparameters inputted by hand.
        :return: No explicit returns but alters dataset permanently.
        """

        resources = (
            self._data_sources,
            self._sources_specs,
            self._variables
        )
        incomplete_sources = []
        for src, src_spec, vs in zip(*resources):
            if 'has_missing_values' in src_spec:
                incomplete_sources.append(src)

                if src_spec['has_missing_values'] is None:
                    raise ValueError("Attempting to make rectangular an "
                                     "ill defined dataset. All sets containing "
                                     "missing values must have a missing sentinel "
                                     "specified.")

                if 'pad_missing_days' in src_spec:
                    if 'date_format' not in src_spec:
                        raise ValueError("Attempting to pad missing days in a "
                                         "time series with no format date specified"
                                         "is illegal. Please specify a date format.")
                    self._pad_missing_days(src_spec, vs, src)

        for src in incomplete_sources:
            vs = self._variables[self._data_sources.index(src)]
            specs = self._sources_specs[self._data_sources.index(src)]

            ignore_vs = []
            if 'time_series' in specs:
                # Ignore date from casting to real values.
                ignore_vs.append(specs['time_series'])

            missing_sent = specs['has_missing_values']
            # self._cast_variables(src)

            target_variables = filter(lambda f: f not in ignore_vs, vs)
            for variable in target_variables:
                missing_sent = specs['has_missing_values']

                if to_obj and variable in to_obj:
                    us_specified = to_obj[variable]

                    def cast(v):
                        return us_specified(v) if v is not missing_sent else v

                else:

                    def _default_cast(el):
                        return float(el) if el is not missing_sent else el

                    # Use default casting to float
                    cast = _default_cast if not user_default_cast else user_default_cast

                v_i = vs.index(variable)
                for dp in src:
                    dp[v_i] = cast(dp[v_i])

            def _is_complete(datapoint):
                return missing_sent not in datapoint

            full_case_analysis = list(filter(_is_complete, src))

            allowed_features = [i for i in range(0, len(vs)) if vs[i] not in ignore_vs]
            if 'time_series' in specs:
                # Ignore date from casting to real values.
                self.time_series_impute(
                    src=src,
                    full_case_analysis=full_case_analysis,
                    # These arguments are provided by default if not
                    # specified by user
                    missing_sentinel=missing_sent,
                    pad_alg_specs=pad_alg_specs,
                    horizontal_padding_methods=horizontal_padding_methods,
                    vertical_padding_methods=vertical_padding_methods,
                    hyper_parameters=hyper_parameters,
                    allowed_features=allowed_features
                )
            else:
                # pad horizontally only
                NotImplemented

    @staticmethod
    def _get_imputation_models(full_case_analysis: list[list[float]],
                               allowed: list[int],
                               specifics: list[typing.Any]):
        """
        Build the imputation models as required. Rounds of training
        are conducted if the models selected include supervised training
        models.

        :param full_case_analysis: The data containing no missing value.
        :param allowed: The allowed modifiable variable in a data row.
        :param specifics: Specifics of the algorithm.
        :return: both horizontal and vertical imputation callable models.
        """

        pad_s, h_methods, v_methods, hypers = specifics

        def _build_new_from_dict(dic, key, fail_msg):
            try:
                model = dic[key]
                model = model.__new__(model)
                model.__init__(full_case_analysis, allowed, hypers)
            except Exception as runtime_exc:
                raise ValueError(fail_msg + "original exception: "
                                 + runtime_exc.__str__())
            return model

        if 'best' in h_methods:
            NotImplemented
            # h_model = get_best_imputation(...)
        else:
            h_model = _build_new_from_dict(
                {'dae': None,
                 'knn': HorizontalKNNModel,
                 'zero': HorizontalZeroImputationModel
                 }, h_methods,
                "Unrecognized imputation method inside "
                "manually specified imputation methods: "
                " {method} not recognized".format(method=h_methods))

        if 'best' in v_methods:
            NotImplemented
            # h_model = get_best_imputation(...)
        elif 'wa' in v_methods:
            v_model = VerticalWeightedAverageModel(full_case_analysis, allowed, hypers)
        elif 'dae' in v_methods:
            NotImplemented
            # h_model = KnnImputation(full_case_analysis, allowed)
        elif 'zero' in v_methods:
            v_model = VerticalZeroImputationModel(full_case_analysis, allowed, hypers)
        else:
            raise ValueError("Unrecognized imputation method inside "
                             "methods manually specified: {method} not "
                             "recognized".format(method=h_methods))

        return v_model, h_model

    def time_series_impute(self, src: list[typing.Any],
                           full_case_analysis: list[list[float]],
                           missing_sentinel: typing.Any,
                           pad_alg_specs: dict,
                           horizontal_padding_methods: str,
                           vertical_padding_methods: str,
                           hyper_parameters: dict,
                           allowed_features: list[int]
                           ):
        """

        :param allowed_features:
        :param missing_sentinel:
        :param src:
        :param full_case_analysis:
        :param pad_alg_specs:
        :param horizontal_padding_methods:
        :param vertical_padding_methods:
        :param hyper_parameters:
        :return:
        """
        # Horizontally knn, dae, zero | vertically dae, wa, zero

        algorithm_specifics = [
            pad_alg_specs,
            horizontal_padding_methods,
            vertical_padding_methods,
            hyper_parameters
        ]
        v_impute, h_impute = self._get_imputation_models(
            full_case_analysis,
            allowed_features,
            specifics=algorithm_specifics
        )

        def _fetch_user_specified_specific(key, instance, default, err_msg):
            """
            TODO:
            :param key:
            :param instance:
            :param default:
            :param err_msg:
            :return:
            """
            _v = default
            if pad_alg_specs and key in pad_alg_specs:
                _v = pad_alg_specs[key]
                if not isinstance(_v, instance):
                    raise ValueError(err_msg)
            return _v

        threshold = _fetch_user_specified_specific(
            key='missing_density_treshold',
            instance=float, default=DEFAULT_DENSITY_VALUE,
            err_msg="Missing density threshold must be a real value "
                    "ranging from zero to one."
        )

        window = _fetch_user_specified_specific(
            key='window',
            instance=int, default=DEFAULT_VERTICAL_WINDOW_SIZE,  # two weeks
            err_msg="Window size must be a Python int when specified"
                    " explicitly."
        )

        def get_local_missing_density(src: list, point: int):
            return 0.0

        src_len = len(src)
        columns = None
        data_to_impute = [i for i in range(0, src_len) if missing_sentinel in src[i]]
        for d_i in data_to_impute:
            missing_density = get_local_missing_density(src, d_i)
            if missing_density > threshold:
                # Fallback to horizontal imputation
                src[d_i] = h_impute(src[d_i], missing_sentinel)
            else:
                # Do not make expensive vertical vectors if not necessary.
                if columns is None:
                    columns = []
                    for feature in range(0, src_len):
                        if feature in allowed_features:
                            col = [dp[feature] for dp in src]
                        else:
                            col = None
                        columns.append(col)

                def _get_view(column, b_wind, t_wind):
                    return column[d_i - b_wind if d_i - b_wind > 0 else 0:
                                  d_i + t_wind if d_i + t_wind < src_len else -1]

                for allow_i in allowed_features:
                    if src[d_i][allow_i] is missing_sentinel:
                        sliding_window = _get_view(columns[allow_i], window, window)
                        src[d_i][allow_i] = v_impute(sliding_window, missing_sentinel)

        return

    def pad_horizontally(self):
        NotImplemented

    def pad_vertically(self):
        NotImplemented

    def execute(self):
        NotImplemented

    def save_back(self, paths_for_each_source, format_for_each_src):
        """
        Save back padded files into provided paths for each source.

        :param paths_for_each_source: Path for each source provided
        :param format_for_each_src: Format for each source provided
        :return: Saves onto device padded datasets
        """

        for path, format_str, src in zip(
                paths_for_each_source,
                format_for_each_src,
                self._data_sources
        ):
            format_list = FormatParser.multi_split(format_str, ['{', '}'])
            Utils.save_file_from_format(path, format_list, src)

    @property
    def desc(self):
        """
        Get a description of the dataset represented by the object.
        The description can contain both user inputted messages and
        autonomously generated ones based on the variables passed as
        arguments and the size of the dataset.

        :return: a string containing the description of the dataset.
        """
        description = []
        for src, specs in zip(self._data_sources, self._sources_specs):
            if 'load_desc' in specs:
                # If file...
                description.append(specs['load_desc'])
            # else create
            NotImplemented

        description = '\n'.join(description)
        return description

    def __str__(self):
        """ Get description of dataset """
        return self.desc

    def delete(self):
        """ Destroys all memory kept by internal data representation. """
        for i in range(len(self._data_sources)):
            del self._data_sources[i]


def _pad_missing(data_path: str, format_str: str):
    format_list = [k for elem in format_str.split('{') for k in elem.split('}')]
    parsed_data = DataFormatReader.parse_file(data_path, format_list[::2])

    train_data = [[float(el) for el in item[:-1]] for item in parsed_data]
    # Encode categorical values.
    num = {'Iris-setosa': -2, 'Iris-virginica': 2, 'Iris-versicolor': 4}
    for i in range(0, len(train_data)):
        train_data[i].append(
            num[parsed_data[i][-1]]
        )

    noise_samples = 12 * len(train_data)
    random_samples = 4 * len(train_data)
    test_shit = train_data.copy()
    nomod = copy.deepcopy(train_data)
    perfect_train = train_data.copy()

    def _gen_noisy():
        nonlocal train_data
        for _ in range(0, noise_samples):
            # Get zeroing out percentage
            p = random.randint(0, 10) / 10
            el = random.choice(test_shit)

            noise = [e if random.uniform(0, 1) > p else 0.0 for e in el]
            train_data.append(noise)
            perfect_train.append(el)
        for _ in range(0, random_samples):
            noise = np.random.randn(len(train_data[0]))
            el = random.choice(test_shit)

            train_data.append(list(np.array(el) + noise))
            perfect_train.append(el)

    _gen_noisy()

    zipped = list(zip(train_data, perfect_train))
    random.shuffle(zipped)
    train_data, perfect_train = zip(*zipped)
    train_data = list(train_data)
    perfect_train = list(perfect_train)
    print(perfect_train)
    layer_size = len(train_data[0])
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=layer_size),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dropout(rate=0.01),
            tf.keras.layers.Dense(150, activation="relu"),
            tf.keras.layers.Dropout(rate=0.01),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dropout(rate=0.01),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(layer_size),
        ]
    )
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_data, perfect_train)).shuffle(400).batch(32)

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mean_squared_error'])

    model.fit(train_ds, epochs=400)
    model.evaluate(train_data)

    for p in range(3, 10):
        true_val = [el[0] for el in test_shit]
        result = model.predict(test_shit)
        pred_nonzero = [el[0] for el in result]

        for i in range(0, len(test_shit)):
            for j in range(0, len(test_shit[0])):
                if random.randint(1, 10) > p:
                    test_shit[i][j] = 0
        print(test_shit)
        result = model.predict(test_shit)
        what_got_in = [el[0] for el in test_shit]
        predicted_val = [el[0] for el in result]

        plt.plot(pred_nonzero, label='predicted with no zeros', linewidth=0.4)
        plt.plot(predicted_val, label='predicted with zeros', linewidth=0.6, color='magenta')
        plt.plot(true_val, label='reality', linewidth=0.4)
        # plt.plot(what_got_in, label='what got in', linewidth=0.4)
        plt.legend()
        plt.show()
        plt.title(f'Auto encoder imputing with p of err {(10 - p) / 10.0} ')
        test_shit = copy.deepcopy(nomod)


def _knn_impute(data_path: str, format_str: str):
    format_list = [k for elem in format_str.split('{') for k in elem.split('}')]
    parsed_data = DataFormatReader.parse_file(data_path, format_list[::2])

    train_data = [[float(el) for el in item[:-1]] for item in parsed_data]
    # Encode categorical values.
    num = {'Iris-setosa': -2, 'Iris-virginica': 2, 'Iris-versicolor': 4}
    for i in range(0, len(train_data)):
        train_data[i].append(
            num[parsed_data[i][-1]]
        )
    test_shit = copy.deepcopy(train_data)

    def _get_neighbours(x: list[...], k: int):
        dists = [np.linalg.norm(np.array(x) - np.array(i)) for i in train_data]
        epsilon = 1e-5

        def sorting(l1, l2):
            # l1 and l2 has to be numpy arrays
            idx = np.argsort(l1).astype(int)
            return l1[idx], l2[idx]

        dists_sort, index_sort = sorting(np.array(dists), np.arange(0, len(dists)))

        ws = 1 / (np.array(dists_sort[:k]) + epsilon)

        return ws, dists_sort, index_sort

    def _pred(x, k):
        ws, dists, indexes = _get_neighbours(x, k)
        norm = np.sum(ws)

        candidates = np.array(train_data)[indexes[:k]]
        new_x = candidates[0] * ws[0]
        # print("Weights: ", ws)
        # print("Candidates: ", candidates)
        for i in range(1, k):
            new_x += candidates[i] * ws[i]
        new_x /= norm
        # print(norm)
        pred = np.multiply(
            new_x,
            (np.array(x) == 0).astype(int)
        )
        # print(pred,x)
        pred = pred + x
        # print(pred)
        return pred

    # test_shit[0][0] = test_shit[0][1] = 0
    # _pred(test_shit[0], 5)
    prob = 5
    for k in range(5, 70, 10):

        true_val = [el[0] for el in test_shit]
        pred_nonzero = [_pred(el, k)[0] for el in test_shit]

        for i in range(0, len(test_shit)):
            for j in range(0, len(test_shit[0])):
                if random.randint(1, 10) > prob:
                    test_shit[i][j] = 0
        # print(test_shit)
        print(train_data)
        result = [_pred(el, k) for el in test_shit]
        what_got_in = [el[0] for el in test_shit]
        predicted_val = [el[0] for el in result]
        # print(result)

        plt.plot(pred_nonzero, label='predicted with no zeros', linewidth=0.4)
        plt.plot(predicted_val, label='predicted with zeros', linewidth=0.7, color='red')
        plt.plot(true_val, label='reality', linewidth=0.4)
        # plt.plot(what_got_in, label='what got in', linewidth=0.4)
        plt.legend()
        plt.title(f"KNN imputing (k={k}) and zero prob{(10 - prob) / 10.0}")
        plt.show()

        test_shit = copy.deepcopy(train_data)


def _pad_dataset(data_path: str, format_str: str):
    format_list = [k for elem in format_str.split('{') for k in elem.split('}')]
    parsed_data = DataFormatReader.parse_file(data_path, format_list[::2])

    dates = [dp[0] for dp in data_path]

    for dp in parsed_data:
        for i in range(0, len(parsed_data[0])):
            if dp[i] == '':
                dp[i] = 'MISSING'

    full_cases = []
    fc_dates = []
    for line in parsed_data:
        if 'MISSING' not in line:
            full_cases.append([float(line[j]) for j in range(1, len(line) - 1)])
            fc_dates.append(line[0])

    def _weighted_predict(date, upw, dww):
        avail_dates = (dates[0], dates[-1])

    # d0 = date(2008, 8, 18)
    # d1 = date(2008, 9, 26)
    # delta = d1 - d0
    # print(delta.days)

    """
    ctot = 0
    fcase = 0
    statuses = [1]*len(parsed_data)
    for i in range(0, len(parsed_data[0])):
        count_l = [dp[i] for dp in parsed_data]

        for j in range(0, len(statuses)):
            if parsed_data[j][i] == 'MISSING':
                statuses[j] = 0

        count = count_l.count('MISSING')
        print(f"Percentage of missing {names[i]}: {count/len(parsed_data)}")
        ctot += count
        fcase += len(count_l)-count

    print(f"FULL CASE: {statuses.count(1) / len(statuses)}")
    print(f"TOTAL FULL-CASE : {fcase}, {fcase/(len(parsed_data)*len(parsed_data[0]))}")
    print(f"TOTAL MISSING: {ctot}, {ctot/(len(parsed_data)*len(parsed_data[0]))}")
    """


if __name__ == '__main__':
    # _pad_dataset(EXAMPLESROOT + '/River Height/LOZZOLO_giornalieri_2001_2022.csv',
    #              '{Data}{PNove};{PZero};{TMedia};{TMax};{TMin};{Vel};{Raf};{Dur};{Set};{Temp};')

    locale.setlocale(locale.LC_ALL, 'fr_FR')

    k = Dataset()
    k.load_source(
        EXAMPLESROOT + '/Meteo/LOZZOLO_giornalieri_2001_2022.csv',
        '{Data};{PNove};{PZero};{TMedia};{TMax};{TMin};{Vel};{Raf};{Dur};{Set};{Temp};',
        has_missing_values='',
        time_series='Data',
        date_format='{day:02}/{month:02}/{year:04}',
        pad_missing_days=True)

    k.make_rectangular(
        to_obj={'Set': lambda x: 0.0},
        vertical_padding_methods='zero',
        horizontal_padding_methods='knn',
        hyper_parameters={'k': 64},
        user_default_cast=lambda x:
        locale.atof(x) if x != '' else ''
    )

    k.save_back([EXAMPLESROOT + '/Meteo/LOZZOLO_giornalieri_2001_2022Pad.csv'],
                [
                    '{Data};{PNove};{PZero};{TMedia};{TMax};{TMin};{Vel};{Raf};{Dur};{Set};{Temp};'
                ],
                )

    # Bocchetta:
    # '{Data};{PNove};{PZero};{Neve};{NeveS};{NeveAlt};{TempAvg};{TempMax};{TempMin};{VelMedia};{Raffica};{Durata};{Rad};'

    # Cellio
    # '{Data};{PNove};{PZero};{TempAvg};{TempMax};{TempMin};'

    # Carcoforo
    # '{Data};{PNove};{PZero};{TempAvg};{TempMax};{TempMin};'

    # Rima
    # '{Data};{PNove};{PZero};{TempAvg};{TempMax};{TempMin};'

    # Lozzolo
    # '{Data};{PNove};{PZero};{TMedia};{TMax};{TMin};{Vel};{Raf};{Dur};{Set};{Temp};'
