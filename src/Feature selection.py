import typing
import warnings
import Config
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from Padding import FormatParser
from DataOrganizer import DataFormatReader


class SourceFile:
    def __init__(self, path: str):
        """
        Initialize a SourceFile object
        :param path: a path to a source file.
        """
        self._path = path
        self._variables = None
        self._raw = None

    def parse(self, format_string: str):

        seps = FormatParser.get_separators(format_string, ['{', '}'])
        names = FormatParser.get_names(format_string, ['{', '}'])

        self._raw = DataFormatReader.parse_file(self._path, seps)
        self._variables = names

    def get_var(self, var_name):
        if var_name not in self._variables:
            raise NameError("No variable is present in target source file "
                            f"with name {var_name}.")
        v_i = self._variables.index(var_name)
        data_col = [d[v_i] for d in self._raw]
        return data_col

    def __del__(self):
        del self._raw

    def map(self, function: typing.Callable, exclude: list = None):
        if not self._raw:
            warnings.warn("Attempted to map function on an empty source file.")
            return

        params = set(range(0, len(self._variables)))
        if exclude:
            excluded_parameters = [self._variables.index(v) for v in exclude]
            params = params - set(excluded_parameters)
        for i in range(len(self._raw)):
            d = self._raw[i]
            self._raw[i] = [
                function(d[k]) if k in params else
                d[k] for k in range(0, len(self._variables))
            ]
        return

    def make_wtccl(self, first: typing.Union[list[float], str],
                   second: typing.Union[list[float], str],
                   user_params: dict = None,
                   filter_1=None, filter_2=None):

        """ Display windowed time-lagged cross correlation graph between
        variables first, second passed as parameters.
        Both first and second can either be a string, representing the label
         of a variable contained inside the source file, or a vector, in which
        case it is taken literally and used in the computation.

        Note that first and second must be of the same size in order to
        compute their correlation. If their size differs, and no other
        measure is indicated (e.g. aligning), the longest between the two
        is truncated down to the size of the smallest one.

        Optional filters on the indicated data can be specified. The
        filters are applied elements-wise on each data point of each vector.

        :param first: The first target variable
        :param second: The second target variable
        :param filter_1: The filter to apply to the first variable. Default
            to None.
        :param filter_2: The filter to apply to the second variable. Defaults
            to None.
        :param user_params: Parameters regarding the graphical display.
            - Splits: controls the number of splits (e.g. each window)
            - Lags: controls the interval of lags to analyze.
            - ColorMap: controls the color map of the output.
                 Must be a color map recognized by matplotlib.
            - Interpolation: controls the color map interpolation.
        """

        def _maybe_load_var(var: str):
            if not isinstance(var, (np.ndarray, list)):
                # Either an iterable or a string is expected.
                if not isinstance(var, str):
                    raise ValueError("Expected variable name or iterable, got "
                                     f"{type(var)} instead. Please provide a "
                                     f"vectorial data row or a variable name.")
                var = self.get_var(var)
            return var

        v_1 = _maybe_load_var(first)
        v_2 = _maybe_load_var(second)

        if np.ndim(v_1) > 1 or np.ndim(v_2) > 1:
            raise ValueError("Expected 1-dim variables {first} and {second}, "
                             "found n-dims. Please reduce the outer dimension "
                             "of the variables in order compute their "
                             "correlation.")

        if np.size(v_2) != np.size(v_1):
            # TODO: alignment...
            if np.size(v_1) > np.size(v_2):
                v_1 = v_1[:np.size(v_2)]
            else:
                v_2 = v_2[:np.size(v_1)]

        if filter_1 is not None:
            v_1 = list(filter(filter_1, v_1))
        if filter_2 is not None:
            v_2 = list(filter(filter_1, v_2))

        params = {
            'Splits': 40,
            'Lags': 7,
            'ColorMap': 'PRGn',
            'Interpolation': 'nearest',
            # ...
        }
        if user_params:
            unrecognized_keywords = set(user_params.keys()) - params.keys()
            if unrecognized_keywords:
                raise ValueError("Unrecognized key arguments passed as parameters: "
                                 f"{unrecognized_keywords}. Available arguments are "
                                 f"present in the __docs__ of the function.")
            params.update(user_params)

        k_1 = np.array_split(v_1, params['Splits'])
        k_2 = np.array_split(v_2, params['Splits'])
        total = []
        lag_range = range(-params['Lags'], +params['Lags'])
        for b_1, b_2 in zip(k_1, k_2):
            c_effs = []
            for lag in lag_range:
                b_1 = np.roll(b_1, lag)
                c_effs.append(np.corrcoef(b_2, b_1)[0][1])
            total.append(c_effs)

        ax = plt.subplot()
        im = ax.imshow(np.array(total),
                       cmap=params['ColorMap'],
                       interpolation=params['Interpolation'],
                       )

        # Put Color bar ont he right
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_xticks(range(0, 2 * params['Lags'], 3))
        ax.set_xticklabels(lag_range[::3])
        ax.set(title='Rolling Windowed Time Lagged Cross Correlation',
               xlim=[0, 2 * params['Lags']],
               xlabel='Offset',
               ylabel='Epochs')
        plt.colorbar(im, cax=cax)
        plt.show()


if __name__ == '__main__':
    source = SourceFile(Config.EXAMPLESROOT + '/Meteo/BOCCHETTA_DELLE_PISSE_giornalieri_1988_2022Pad.csv')
    source.parse(
        '{DataBo};{PNoveBo};{PZeroBo};{NeveBo};{NeveSBo};{NeveAltBo};{TempAvgBo};{TempMaxBo};{TempMinBo};{VelMediaBo};{RafficaBo};{DurataBo};{RadBo};')

    source.map(lambda x: float(x), exclude=['DataBo'])
    source.make_wtccl('PZeroBo', 'PNoveBo', user_params={'Lags': 14, 'Splits': 40, 'ColorMap': 'bwr'})

    river_source = SourceFile(Config.EXAMPLESROOT + '/River Height/sesia-hourly-packed-padded-aligned.csv')
    river_source.parse('{Data};{Vec}')


    def _get_vec(x, enclosed=True):
        v = np.array(np.fromstring(x[1:-1].strip() if enclosed else x, float, sep=', '))
        return np.mean(v)


    data = source.get_var('DataBo')
    rdata = river_source.get_var('Data')
    i = data.index('23/03/2018')

    dates = data[i:]
    print(dates)
    i_2 = rdata.index('31/12/2022')

    dates_river = rdata[:i_2 + 1]
    for d in dates:
        if not d in dates_river:
            print(f'{d} IS MISSING!!')

    print("RIPETUTE:", len(set(dates_river)) - len(dates_river),
          len(set(dates)) - len(dates))
    a = source.get_var('PZeroBo')
    meteo = a[i:]

    river_source.map(_get_vec, exclude=['Data'])
    river = river_source.get_var('Vec')[:i_2 + 1]
    std = np.std(river)
    mean = np.mean(river)
    river = (river - mean) / std
    print(len(river), len(meteo))

    river_source.make_wtccl(river, meteo,
                            user_params={'Lags': 14,
                                         'Splits': 30,
                                         'ColorMap': 'PRGn'
                                         }
                            )
    print(river_source.get_var('Vec'))
