"""
Support module to request satellite data from the NASA sat-data API.

LICENCE: NASA does not indemnify nor hold harmless users of NASA material,
 nor release such users from copyright infringement, nor grant exclusive
 use rights with respect to NASA material.

NASA material is not protected by copyright within the United States, unless noted.
If copyrighted, permission should be obtained from the copyright owner prior to use.
If not copyrighted, NASA material may be reproduced and distributed without further
 permission from NASA.

 """
import re
import shutil
import requests

import email
import imaplib
import base64

from PIL import Image
from typing import Tuple
from os import path
from datetime import date
from dateutil.rrule import rrule, DAILY

import Config

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


# I simboli che l'API arpa associa alle grandezze richieste
ARPA_VALUES = {
    'Altezza idrometrica': 'I',
    'Precipitazione giornaliera': 'P',
    'Temperatura aria': 'T',
    'Gradi giorno': 'K',
    'Vento': 'V',
    'Neve': 'N',
    'Radiazioni': 'R',
    'Umidità': 'H'
}

# Degli alias per comodità.
ALIASES = {
    'Altezza idrometrica': ['Altezza', 'Alt', 'Height'],
    'Precipitazione giornaliera': ['Precipitazioni', 'Prec', 'Rain'],
    'Temperatura aria': ['Temperatura', 'Temp', 'Temperature'],
    'Gradi giorno': ['Gradi', 'Grad', 'Degrees'],
    'Vento': ['Vent', 'Wind'],
    'Neve': ['Nev', 'Snow'],
    'Radiazioni': ['Rad', 'Rads'],
    'Umidità': ['Hum', 'Um', 'Humidity']
}

ARPA_FORM = 'https://www.arpa.piemonte.it/radar/open-scripts/richiesta_dati_gg.php'
API_EMAIL = 'ggriverflow1@outlook.it'
PASSWORD = 'Riverflow'


def arpa_data():
    """

    :return:
    """

    def _feature_labels_to_request(features: list[str]):
        sym_features = ''
        for feature in features:
            if feature not in ARPA_VALUES.keys():
                # Verifica che un alias sia presente con lo stesso nome.
                alias = next(iter(
                    [k for k in ALIASES if feature in ALIASES[k]]),
                    None
                )
                if alias is None:
                    raise ValueError("Una delle grandezze non è stata riconosciuta e "
                                     "non è presente nel registro degli alias. Perfavore,"
                                     f"verifica che la richiesta di '{feature}' sia "
                                     f"corretta!")
                feature = alias
            sym_features += ARPA_VALUES[feature]
        return sym_features

    imap_server = "imap-mail.outlook.com"
    imap_port = 993
    imap = imaplib.IMAP4_SSL(imap_server, port=imap_port)
    try:
        imap.login(API_EMAIL, PASSWORD)
    except imaplib.IMAP4.error:
        raise ValueError("ARPA API misconfigured. Please provide a correct "
                         "IMAP mail repository. (Login failed, incorrect "
                         " password attempt?)")
    status, messages = imap.select('INBOX')
    messages = int(messages[0])

    def _verify_sender_identity(mail):
        try:
            _sender = mail.get('From')
            _subject = mail.get('Subject')
            _req_tok = 'RICHIESTA DATI RETE METEOIDROGRAFICA ARPA PIEMONTE'
            if _sender != 'virtcsi-iris@arpa.piemonte.it' or _subject != _req_tok:
                raise ValueError
        except (KeyError, ValueError):
            return False
        return True

    def _get_part_file_specs(part: email.message.Message):
        """ Ottieni il nome e l'estensione del segmento di mail passato come
        argomento, se presenti. Altrimenti ritorna il tuple None, None. """
        f_name = str(part.get_filename())
        f_ext = re.compile(r'.*\.(?P<extension>.+)').findall(f_name)

        return f_name, f_ext

    def _get_response_id(email):
        r_id = re.compile(r'richiesta_dati_gg_(?P<response_id>[0-9]+)')
        for possible_attached in email.walk():
            f_name, _ = _get_part_file_specs(possible_attached)
            found_id = r_id.findall(f_name)
            if not found_id:  # Prova a parsare ogni file in cerca dell'id.
                continue
            elif len(found_id) > 1:
                import warnings
                warnings.warn(
                    "Più di un id è stato in una singola risposta "
                    "ad una richiesta. Questo può significare sia un errore "
                    "da parte dell'API sia un tentativo di ingannare il "
                    "parser di un entità esterna. Il parsing continuerà "
                    "comunque. Si prega di verificare manualmente le mail.",
                    category=RuntimeWarning
                )
            try:
                req_id = int(found_id[0])  # Se più di un id è presente, prendi il primo.
            except (ValueError, IndexError):
                raise ValueError("La richiesta api è fallita in quanto la risposta "
                                 "presenta un id scorretto. Perfavore, verifica manualmente "
                                 "le mail.")

            return req_id  # Se nulla dà errore, fidati e ritorna il primo id.
        # Se nessuna parte ha un id valido allora la risposta non ha un id.
        return None

    typ, data = imap.search(None, '(UNSEEN)')
    print("Data: ", data)
    typ, data = imap.fetch('4', '(RFC822)')
    # noinspection PyUnresolvedReferences
    raw_email = data[0][1]
    print(typ)

    print(raw_email)
    raw_email_string = raw_email.decode('utf-8')
    email_message = email.message_from_string(raw_email_string)
    print("Verify: ", _verify_sender_identity(email_message))

    def _get_content_transfer_encoding(email_part: email.message.Message):
        """ Ottieni il tipo di encoding dell'allegato presente nella mail """
        if 'Content-Transfer-Encoding' in email_part.keys():
            return email_part.get('Content-Transfer-Encoding')
        else:
            return None

    print("L'id è :", _get_response_id(email_message))
    # get_csv_from_imap_mail()
    for part in email_message.walk():
        # print(part)
        # Cast to string to avoid None
        file_name = str(part.get_filename())
        file_extension = re.compile(r'.*\.(?P<extension>.+)').findall(file_name)
        if not file_extension or 'csv' not in file_extension:
            # Do not attempt parsing
            continue

        if part.get_content_maintype() == 'application':
            if _get_content_transfer_encoding(email_part=part) != 'base64':
                raise ValueError("CSV format attached was encoded in an unknown "
                                 "format. Please use base64 instead.")

            raw_csv_file = base64.b64decode(str(part.get_payload()))
            csv_file = raw_csv_file.decode('utf-8')
            print(csv_file)
            # Mark email as read.
            imap.store('2', '+FLAGS', '\\Seen')
        elif part.get_content_maintype() == 'text':
            print("Textual")
            print(part.get_payload())

    """
    table_name = 'richiesta_dati.dati_giornalieri'
    req_url = 'https://www.arpa.piemonte.it/radar/open-scripts/richiesta_dati_gg.php?richiesta=1'

    req_data = {
        'data': '{"email":"zanilorenzopm@gmail.com","data_inizio":"2023-03-24","data_fine":"2023-04-24","richiedente":"4","tipofile":"pdf","parametri":["I"],"stazioni":["379"]}',
        'dest_table':table_name
    }
    response = requests.post(req_url, data=req_data)
    print(response)
    try:
        print(response.content)
    except:
        pass
    try:
        print(response.raw)
    except:
        pass
    """
    """
    mail = 'zanilorenzopm@gmail.com'
    browser = mechanicalsoup.Browser()
    form_page = browser.get(ARPA_FORM)
    html = form_page.soup

    captcha_decl = re.compile('[ \t\n]*var [xy] = (?P<value>[0-9]+);')
    expressions = captcha_decl.findall(html.__str__())

    captcha_solution = sum([int(v) for v in expressions])

    feature_requested_form = html.select('#form_stazione')[0]
    feature_requested_form.select('#filterParametro')[0]['value'] = 'Vento'
    feature_requested_form.select('#filterSiglaPro')[0]['value'] = 'Asti'
    feature_requested_form.select('#stazioni_filtrate')[0]['value'] = 'ASTI'

    browser.submit(feature_requested_form, ARPA_FORM)
    # Get form to fill
    form_to_submit = html.select('#form1')[0]
    print(form_to_submit.s)
    # Fill email
    form_to_submit.select('#mail')[0]['value'] = mail
    # Fill confirmation email
    form_to_submit.select('#conf_mail')[0]['value'] = mail
    # Fill captcha
    form_to_submit.select('#calcolo_antispam')[0] = str(captcha_solution)

    pdf_check = form_to_submit.select('#tipofile')[0]
    csv_check = form_to_submit.select('#tipofile')[1]

    # Select csv values

    form_to_submit.select('#datadal')[0]['value'] = '2021-03-21'
    form_to_submit.select('#dataal')[0]['value'] = '2021-03-21'
    browser.launch_browser(soup=html)
    """


if __name__ == '__main__':
    arpa_data()
