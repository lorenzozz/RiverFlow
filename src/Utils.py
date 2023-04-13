from DataOrganizer import DataFormatReader  # Csv naive parsing
from Config import *


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

    with open(dest_path, "w") as new_csv_file:
        # Get indices of all labels in original formatting string
        indices = [f_list.index(var) for var in f_list[1::2]]

        lines = []
        for p in sampled_points:
            for i, val in zip(indices, p):
                f_list[i] = str(val)
            replace = ''.join(f_list) + '\n'
            lines.append(replace)

        new_csv_file.writelines(lines)


if __name__ == '__main__':
    _sample_n_for_each_hour(RIVERDATAROOT + '/sesia-height.csv', RIVERDATAROOT + '/sesia-hourly.csv',
                            '{Date} {Hour}:{Garbage};{Height}', 1)
