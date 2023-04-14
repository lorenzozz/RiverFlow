import shutil
import requests

from PIL import Image
from typing import Tuple
from os import path
from datetime import date
from dateutil.rrule import rrule, DAILY

import Config

"""
Support module to request satellite data from the NASA sat-data API
"""

REQ_TIME = '2023-03-24T00:00:00'
SAT_REQ_URL = 'https://gibs-c.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?'
SAT_TILE_INFO = 'Z&layer=IMERG_Precipitation_Rate&style=default&tilematrixset=2km&Service=WMTS&Request=GetTile' \
                '&Version=1.0.0&Format=image%2Fpng&TileMatrix=2&TileCol=2&TileRow=0'


def get_sat_img(year: int, month: int, day: int, flush_cache: bool = False) -> str:
    """
    Returns a path to a satellite image from the specified date.
    If flush is specified to be true, regardless of the cached value a new satellite
    image is fetched from the specified API.
    Otherwise, if the requested image is already present in the folder, return
    that address and issue no request.

    :param: year: Sample year.
    :param: month: Sample month.
    :param: day: Sample day.
    :param: flush_cache: Controls cache management. See above.
    :return: A path to a satellite image of the specified date
    """

    file_v_time = f'{year:04d}-{month:02d}-{day:02d}'
    req_time = f'{file_v_time}T00:00:00'
    gen_path = Config.RIVERDATAROOT + f'/SatelliteData/SAT-DATA-{file_v_time}.png'

    # Request data from nasa api only if requested data is not already cached or
    # if a flush to cache has been requested.
    if not path.exists(gen_path) or flush_cache:

        response = requests.get(SAT_REQ_URL + 'TIME=' + req_time + SAT_TILE_INFO, stream=True)
        response.raw.decode_content = True

        with open(gen_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)

        # Explicitly delete unused memory
        del response

        # Overwrite precedent incomplete image
        b = Image.open(gen_path).convert('RGBA')
        local_path = Config.RIVERDATAROOT + '/eu-landscape.png'
        c = Image.open(local_path).convert('RGBA')

        # Merge geographic info and precipitation landscape together
        b.alpha_composite(c)

        b.save(gen_path)

        del b
        del c

    return gen_path


def populate_range(years: Tuple[int, int], months: Tuple[int, int], days: Tuple[int, int],
                   verbose: bool = False, flush_cache: bool = False) -> None:
    """
    Populate the requested folder with samples from satellite data acquired from the http api
    specified in global parameters SAT_REQ_URL, SAT_TILE_INFO.
    :param flush_cache: Signal whether to accept cached data or not. defaults to Do not flush
    :param years: Range of years to sample
    :param months: Range of months to sample
    :param days: Range of days to sample
    :param verbose: Output task done on console or don't
    :return: No explicit return, fills RiverData folder with satellite data.
    """

    a = date(years[0], months[0], days[0])
    b = date(years[1], months[1], days[1])

    for time_date in rrule(DAILY, dtstart=a, until=b):

        req_time = f'{time_date.year:04d}-{time_date.month:02d}-{time_date.day:02d}T00:00:00'

        get_sat_img(time_date.year, time_date.month, time_date.day, flush_cache=flush_cache)

        if verbose:
            print(f"* Acquired satellite data from {req_time}")


if __name__ == '__main__':
    populate_range((2022, 2022), (1, 2), (1, 15), verbose=True, flush_cache=True)
