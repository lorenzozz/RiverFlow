import numpy as np
from typing import Iterable, Type

from datetime import datetime
from dateutil.rrule import rrule, DAILY
from Config import *
from DataOrganizer import DataFormatReader  # Csv naive parsing


def _save_file_from_format(dest_path: str, format_list: list, data_points: Iterable):
    """
    Saves data points given in data_points inside the destination path specified according
    to the format list provided. Expects data in row form, not column-wise. (c-contiguous)
    :param data_points: Data to save
    :param dest_path: Destination file path
    :param format_list: Format list
    :return: No explicit return, saves file into dest_path
    """
    with open(dest_path, "w") as new_csv_file:
        # Get indices of all labels in original formatting string
        indices = [format_list.index(var) for var in format_list[1::2]]

        lines = []
        for p in data_points:
            for i, val in zip(indices, p):
                format_list[i] = str(val)
            replace = ''.join(format_list) + '\n'
            lines.append(replace)

        new_csv_file.writelines(lines)


def _sample_n_for_each_hour(path: str, dest_path: str, format_str: str, n: int):
    """
    Samples n elements from each day from csv file taken at location path.
    Failing to meet the expectation of n elements is not a fatal error, but
    'false' data will be added as padding from the most recent data sample.
    If no data is present for a day at all an error is issued. (Not enforced)
    Expects at least one of the fields in format_str must be 'Hour' as compare field.

    :param path: The source file where data will be taken
    :param dest_path: The destination file where data will be saved into
    :param format_str: A format string describing the formatting of the line string
    :param n: The amount of samples
    """

    if path is None or dest_path is None:
        return

    f_list = [a for arg in format_str.split('{') for a in arg.split('}')]
    data = DataFormatReader.parse_file(path, f_list[::2])

    hour_i = f_list[1::2].index('Hour')

    sampled_points = []
    n_p = l_entry = 0
    curr_hour = data[0][hour_i]
    for d_point in data:
        if d_point[hour_i] != curr_hour:
            # Put 'false' data to fill samples requested
            if l_entry:
                while n_p < n:
                    sampled_points.append(l_entry)
                    n_p += 1
            n_p = l_entry = 0
            curr_hour = d_point[hour_i]

        if n_p < n:
            n_p += 1
            l_entry = d_point
            sampled_points.append(d_point)

    _save_file_from_format(dest_path, format_list=f_list, data_points=sampled_points)


def _pack_daily(path: str, dest_path: str, format_str: str, target_var: str, cast_to: Type, verbose: bool = False):
    """
    Pack together all datapoints belonging to a same day. 24 points are expected for
    each day. Failing to meet this criterion leads to a fatal error (NotImplemented)
    Note: implicitly expects 'Hour' to be inside the format string.
    Also expects 'Date' to be inside the format string.

    Moreover, 00:00 must be interpreted to belong to the next day in the data
    e.g. march 10, 23:59 -> march 11, 00:00

    :param: target_var: Target variable to pack
    :param: path: Target path location
    :param: dest_path: Save file path location
    :param: format_str: Format string of the target csv file
    :param: cast_to: Type of target variable
    :param: verbose: Print incomplete days
    :return: No explicit return, saves packed data inside save path file
    """

    form_l = [el for sp in format_str.split('}') for el in sp.split('{')]
    data = DataFormatReader.parse_file(path, form_l[::2])

    if target_var not in form_l or 'Hour' not in form_l or 'Date' not in form_l:
        raise Exception("Incorrect call to _pack_daily(). Hourly data required with target variable.")

    hour_i, date_i, targ_i = form_l[1::2].index('Hour'), form_l[1::2].index('Date'), form_l[1::2].index(target_var)

    # Find first and last aligned datapoints, glance over isolated data samples
    # Expects military date format HH:MM:SS
    trim_data = data[next(j for j in range(0, len(data)) if data[j][hour_i] == '00'):
                     -next(j for j in range(0, len(data)) if data[-j][hour_i] == '00')]

    # Takes list by reference and pads missing hourly values.
    def pad_missing_into_day(day: list, time_series: list):

        # No way to pad extreme edge cases intelligently.
        if len(day) < 4:
            day += [cast_to(0.0)] * (24 - len(day))
            return

        day_series = {i for i in range(0, 24)}
        missing = sorted(list(day_series.difference(set([int(day[hour_i]) for day in time_series]))))

        for missing_hour in missing:
            sub_val = np.mean(np.array(day).astype(np.float32))
            day.insert(missing_hour, sub_val)

    # Cannot assign both to [] as only one reference is created
    grouped_data = []
    curr_day_data = []
    curr_date = trim_data[0][date_i]
    for d_p in trim_data:
        # New day
        if d_p[date_i] != curr_date:
            # Missing datapoints
            if len(curr_day_data) != 24:
                if verbose:
                    print(f"> Missing datapoint at date {curr_date}")

                past_date_ind = [date[date_i] for date in trim_data].index(curr_date)
                pad_missing_into_day(curr_day_data, trim_data[past_date_ind:past_date_ind+len(curr_day_data)])

            grouped_data.append([curr_date, curr_day_data])
            curr_date = d_p[date_i]
            curr_day_data = []

        curr_day_data.append(cast_to(d_p[targ_i]))

    # If last datapoint is malformed, just glance over it
    if len(curr_day_data) == 24:
        grouped_data.append([curr_date, curr_day_data])

    print([len(g[1]) for g in grouped_data].count(24), len(grouped_data))
    _save_file_from_format(dest_path, ['', 'Date', ';', target_var, ''], grouped_data)


def _check_series_for_missing_days(data: list, date_format: str) -> list:
    """

    :param data:
    :param date_format:
    :return:
    """
    a = datetime.strptime(data[0], '%Y-%m-%d')
    b = datetime.strptime(data[-1], '%Y-%m-%d')

    missing = []
    for time_date in rrule(DAILY, dtstart=a, until=b):
        # Get representation of day according to
        # date format
        day_rep = date_format.format(
                day=time_date.day,
                month=time_date.month,
                year=time_date.year
        )

        if day_rep not in data:
            missing.append(time_date)

    return missing


def _check_file_for_missing_days(file_path: str, format_str: str, date_format: str, verbose: bool = False) -> list:
    """
    Checks for missing days inside a time series. Expects at least
    one field inside the format string to be labeled 'Date'.
    Returns a list containing all missing days in a series. If none are missing,
    an empty list is returned instead.

    :param file_path: A path to the file to be checked
    :param format_str: A string representing the formatting of file requested
    :param verbose: Whether to print to output missing days found.
    :param date_format: A string to indicate the date format, for example
        {day}{month}{year}
    :return: whether the series contains missing days
    """

    format_list = [k for elem in format_str.split('{') for k in elem.split('}')]
    parsed_data = DataFormatReader.parse_file(file_path, format_list[::2])

    if 'Date' not in format_list[1::2]:
        raise NameError("Expected one of the variables referenced in the format "
                        "string to be labeled as 'Date' for internal use, got "
                        "no such variable instead.")

    date_i = format_list[1::2].index('Date')
    missing_days = _check_series_for_missing_days(
        [
            # For each data point, get the corresponding date as
            # expressed in the format string.
            data_point[date_i] for data_point in parsed_data
        ], date_format
    )
    if verbose:
        for missing_day in missing_days:
            print(f"> Found missing day in file, {missing_day}")

    return missing_days


if __name__ == '__main__':

    _check_file_for_missing_days(EXAMPLESROOT + '/River Height/sesia-hourly-packed.csv',
                                 '{Date};{Garbage}', '{year:04}-{month:02}-{day:02}', verbose=True)
