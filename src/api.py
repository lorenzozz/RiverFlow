import shutil
from typing import Tuple

import Config
import requests
from PIL import Image

"""
Support module to request satellite data from the NASA sat-data API
"""

REQ_TIME = '2023-03-24T00:00:00'
SAT_REQ_URL = 'https://gibs-c.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?'
SAT_TILE_INFO = 'Z&layer=IMERG_Precipitation_Rate&style=default&tilematrixset=2km&Service=WMTS&Request=GetTile' \
                '&Version=1.0.0&Format=image%2Fpng&TileMatrix=2&TileCol=2&TileRow=0'


def populate_range(years: Tuple[int, int], months: Tuple[int, int], days: Tuple[int, int], verbose=False):
    """
    Populate the requested folder with samples from satellite data acquired from the http api
    specified in global parameters SAT_REQ_URL, SAT_TILE_INFO.
    :param years: Range of years to sample
    :param months: Range of months to sample
    :param days: Range of days to sample
    :return: No explicit return, fills RiverData folder with satellite data.
    """

    # Complete range
    def complete_range(tup):
        return range(tup[0], tup[1] + 1)

    for year in complete_range(years):
        for month in complete_range(months):
            for day in complete_range(days):
                file_v_time = f'{year:04d}-{month:02d}-{day:02d}'
                req_time = f'{file_v_time}T00:00:00'
                gen_path = Config.RIVERDATAROOT + f'/SAT-DATA-{file_v_time}.png'

                # Request data from nasa api
                response = requests.get(SAT_REQ_URL + 'TIME=' + req_time + SAT_TILE_INFO, stream=True)
                response.raw.decode_content = True

                with open(gen_path, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)

                # Explicitly delete unused memory
                del response

                # Overwrite precedent incomplete image
                b = Image.open(gen_path).convert('RGBA')
                l = Config.RIVERDATAROOT + '/eu-landscape.png'
                c = Image.open(l).convert('RGBA')

                # Merge geographic info and precipitation landscape together
                b.alpha_composite(c)

                b.save(gen_path)
                if verbose:
                    print(f"* Got satellite data from {req_time}")
                del b
                del c


if __name__ == '__main__':
    populate_range((2022, 2022), (1, 2), (1, 15), True)
