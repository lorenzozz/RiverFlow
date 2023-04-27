import copy
import random
import typing
import warnings
import numpy as np
import tensorflow as tf
import Utils

from Config import *
from DataOrganizer import DataFormatReader
from matplotlib import pyplot as plt
from typing import Iterable


class FormatParser:
    """
    Namespace for both string format parsing routines
    and plan description parsing routines.
    No explicit initialization is required as all methods are
    class methods.
    """

    @classmethod
    def _multi_split(cls, string, splits: Iterable):
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
        return cls._multi_split(string, splits)[::2]

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
        return cls._multi_split(string, splits)[1::2]


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
        NotImplemented

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
        _variables and _data_sources are not altered during the flush
         as they can only be set once.
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
                         horiz: str = 'best', vert: str = 'best',
                         hyper: dict = None,
                         us_def_cast : typing.Callable = None):
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

        :param pad_alg_specs: The specific of the imputation algorithm as in the docs.
        :param to_obj: The routines mapping strings to the objects. Defaults to
            casting to float.
        :param horiz: The horizontal imputation methods applied.
        :param vert: The vertical imputation methods applied.
        :param hyper: The optional hyperparameters inputted by hand.
        :return: No explicit returns but alters dataset permanently.
        """

        incomplete_srcs = []
        for src, src_spec, vs in zip(
                self._data_sources,
                self._sources_specs,
                self._variables
        ):
            if 'has_missing_values' in src_spec:
                incomplete_srcs.append(src)

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

        def _default_cast(el):
            return float(el)

        _def_cast = None
        if not us_def_cast:
            _def_cast = _default_cast
        else:
            _def_cast = us_def_cast

        for src in incomplete_srcs:
            vs = self._variables[self._data_sources.index(src)]
            specs = self._sources_specs[self._data_sources.index(src)]

            ignore_vs = []
            if 'time_series' in specs:
                # Ignore date from casting to real values.
                ignore_vs.append(specs['time_series'])

            for variable in filter(lambda f: f not in ignore_vs, vs):
                # User has specified a custom conversion
                if variable in to_obj:
                    cast = to_obj[variable]
                else:
                    # Else use default casting to float.
                    cast = _def_cast

                v_i = vs.index(variable)
                print(v_i)
                map(lambda dp: cast(dp[v_i]), src)
        print(incomplete_srcs[:5])

    def pad_horizontally(self):
        NotImplemented

    def pad_vertically(self):
        NotImplemented

    def execute(self):
        NotImplemented

    @property
    def desc(self):
        """

        :return:
        """
        description = []
        for src, specs in zip(self._data_sources, self._sources_specs):
            if 'load_desc' in specs:
                # If file...
                description.append(specs['load_desc'])
            # else create
        description = '\n'.join(description)
        return description

    def __str__(self):

        description = self.desc
        if description:
            return description
        else:
            return None

    def delete(self):
        """
        Destroys all memory kept by internal data representation.
        """
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

    k = Dataset()
    k.load_source(EXAMPLESROOT + '/River Height/LOZZOLO_giornalieri_2001_2022.csv',
                  '{Data};{PNove};{PZero};{TMedia};{TMax};{TMin};{Vel};{Raf};{Dur};{Set};{Temp};',
                  has_missing_values='',
                  time_series='Data',
                  date_format='{day:02}/{month:02}/{year:04}',
                  pad_missing_days=True)
    k.make_rectangular(to_obj={
        'Temp': lambda label: {'NE': 3, 'NNE': 4, 'SE': 5,
                               'SW': 6, 'N': 7, 'S': 8, 'SSW': 9,
                               'NNW': 10, 'SSE': 11}[label]}
    )
