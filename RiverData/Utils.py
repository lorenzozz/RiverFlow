
def _sample_n_for_each_day(path: str, dest_path:str, format:str):
    """
    Samples n elements from each day from csv file taken at location path.
    Failing to meet the expectation of n elements is not a fatal error, but
    'false' data will be added as padding from the most recent data sample.
     If no data is present for a day at all an error is issued.

    :param path: The source file where data will be taken
    :param dest_path: The destination file where data will be saved into
    :param format: A format string describing the formatting of the line string
    """

    if path is None or dest_path is None:
        return

    sampled_lines = []
    with open(path, 'r') as inp_file:
        lines = inp_file.readlines()



    with open(dest_path, 'w') as save_file:
        save_file.writelines(dest_path)
