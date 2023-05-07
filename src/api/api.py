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
import warnings
import requests
import email
import imaplib
import base64
import threading

from typing import Tuple
from os import path
from datetime import date
from dateutil.rrule import rrule, DAILY

from RiverFlow.src.api import apierrors

try:
    from PIL import Image
except ImportError:
    raise ImportError(
        "Non è stato possibile importare la libreria richiesta per la "
        "modifica dei file immagine. Perfavore, verifica di avere installato "
        "PIL sul tuo pip."
    )

import RiverFlow.src.Config as Config

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


"""
    - I simboli definiti in ARPA_VALUES sono quelli riconosciuti dalla API -
    - meteo dell'ARPA. Nessun altro simbolo è disponibile                  -
    > Da ARPA-Richiesta Radar-Legenda.pdf:
    
    - Precipitazione
          Indica l’altezza della pioggia caduta e dell’equivalente in acqua delle 
        precipitazioni solide (mm). Il dato fornito (totale) corrisponde al valore cumulato 
        nelle 24 ore e, per convenzione, viene attribuito al giorno successivo.
          Ad esempio la precipitazione del giorno 15 ottobre ha convenzionalmente la
        data del 16 ottobre.
        
    - Temperatura dell’aria
          Valori massimi e minimi giornalieri rispetto a tutti i dati registrati e 
        la media giornaliera calcolata come media di tutti i valori registrati nel giorno (°C).
    
    - Vento 
          [...] i valori della velocità della massima raffica, cioè il massimo valore 
        dei campionamenti effettuati nel giorno; la direzione di provenienza della 
        massima raffica, espressa in gradi sessagesimali a partire da 0°(nord) e poi 
        ruotando in senso orario [...] il valore medio dei dati registrati nel giorno.
    
    - Livello idrometrico
          Valori massimi e minimi giornalieri rispetto a tutti i dati registrati e la 
        media giornaliera calcolata come media di tutti i valori registrati nel giorno (m).
          
    - Neve
          Neve al suolo: valore relativo all’altezza della neve al suolo (cm), 
        registrato alle ore 8:00 locali.
          Neve fresca: valore relativo all’altezza dell’accumulo di nuova neve (cm) 
        registrato nelle 24 ore precedenti (dalle 8:00 alle 8:00 locali).

    - Radiazione solare globale
          Valore giornaliero della radiazione diretta e della radiazione globale 
        diffusa, nell’unità di superficie orizzontale, calcolato come integrale 
        dalle 0 alle 24 dei valori registrati
    
    - Umidità relativa dell’aria
          Si forniscono i valori massimi e minimi giornalieri rispetto a tutti i dati 
        campionati e la media di tutti i valori registrati nel giorno. I valori forniti 
        rappresentano il rapporto tra la quantità di vapor d'acqua effettivamente 
        presente nella massa d'aria e la quantità massima che essa può contenere a quella
        temperatura e pressione (%).
    
    - Gradi Giorno (GG)
          Corrispondono alla sommatoria della differenza tra la temperatura di 
        riferimento (Trif =20°C) e la temperatura media giornaliera, calcolata 
        solo per i contributi positivi.

"""

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

"""
  -  Definiti altri alias per comodità di utilizzo. Ogni alias ha uno e un solo -
  -  simbolo associato.                                                         -
"""

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

# TODO: Wrap into github gitignore
ARPA_FORM = 'https://www.arpa.piemonte.it/radar/open-scripts/richiesta_dati_gg.php'
API_EMAIL = 'ggriverflow1@outlook.it'
PASSWORD = 'Riverflow'

_MAX_API_ARPA_REQUESTS = 5


def _feature_labels_to_request(features: list[str]):
    """ Converti le grandezzi richieste nei codici dell'API dell'ARPA.
    Qunado una grandezza non viene riconosciuta immediatamente, viene fatto
    un controllo per verificare se appare negli alias. Se non appare, viene
    fatto il raise di un errore. """
    sym_features = ''
    for feature in features:
        if feature not in ARPA_VALUES.keys():
            # Verifica che un alias sia presente con lo stesso nome.
            possible_alias = next(iter(
                [k for k in ALIASES if feature in ALIASES[k]]),
                None
            )
            if possible_alias is None:
                raise ValueError(
                    "Una delle grandezze non è stata riconosciuta e "
                    "non è presente nel registro degli alias. Perfavore, verifica "
                    "che la richiesta di '{f}' sia corretta!".format(f=feature)
                )
            feature = possible_alias
        sym_features += ARPA_VALUES[feature]
    return sym_features


def _get_part_file_specs(part: email.message.Message):
    """ Ottieni il nome e l'estensione del segmento di mail passato come
    argomento, se presenti. Altrimenti ritorna il tuple None, None. """
    f_name = str(part.get_filename())
    f_ext = re.compile(r'.*\.(?P<extension>.+)').findall(f_name)

    return f_name, f_ext


class IMAPAccesser:
    """ Entità che gestisce l'accesso al server IMAP. """
    _imap_port = 993
    _imap_server_address = 'imap-mail.outlook.com'

    def __init__(self, imap_mail_address: str, expected_ids: list[int]):
        """
        Inizializza un istanza di un IMAPAccesser.

        :param imap_mail_address: Il server IMAP che dovrà gestire.
        :param expected_ids: La lista che contiene tutti gli ID delle richieste
            che il gestore dovrà ricevere. Continuerà ad agire finché tutti gli
            id non saranno ricevuti correttamente.
        """

        self._available_request_ids = dict()
        self._remaining_expected_ids = expected_ids
        self._available_id_lock = threading.Lock()

        if not re.compile(r'.+@.+\..+').findall(imap_mail_address):
            raise ValueError(
                "L'indirizzo email fornito all'API IMAP ha fallito un "
                "semplice test di validità. Perfavore, verifica che "
                "l'indirizzo impostato è corretto per poter procedere."
            )

        self._address = imap_mail_address
        self._imap = None
        self._authenticated = False

    def get_unread_email_indices(self):
        """ Ottieni gli 'indici' del protocollo IMAP di tutte le email non
        segnate come lette attualmente. """
        if not self._imap:
            raise ValueError(
                "Stai cercando di leggere dei messaggi da un server IMAP "
                "che non è stato aperto propriamente tramite il context "
                "manager. Apri la connessione con un context manager prima di "
                "utilizzare questo metodo."
            )
        typ, data = self._imap.search(None, '(UNSEEN)')
        if 'OK' not in typ:
            raise apierrors.RequestError(
                "La richiesta di accedere agli elementi nel server IMAP non "
                "letti ha generato un eccezione. Verifica che il server sia "
                "online e che supporti il protocollo IMAP.",
                failed_request=self.get_unread_email_indices.__name__
            )
        return typ, data

    def maybe_get_email_if_available(self, request_id: int):
        """ Se la mail con l'id richiesto è disponibile, prendila dal
        dizionario con lock che contiene le email prese dalla casella postale. """
        if not isinstance(request_id, int):
            raise ValueError(
                "Un errore interno dell'api ha portato alla generazione di un "
                "id di richiesta incorretto. Per favore, verifica che la richiesta "
                "effettuata il codice sorgente siano corretti."
            )
        with self._available_id_lock:
            if request_id in self._available_request_ids:
                return self._available_request_ids[request_id]
        return None

    def expect_new_request_id(self, request_id):
        """ Aggiungi un nuovo id di richiesta alla lista degli id di cui
        il manager si deve occupare. Può essere chiamato anche prima di
        effettuare l'accesso al server IMAP in un context manager. """
        with self._available_id_lock:
            self._remaining_expected_ids.append(request_id)

    def __enter__(self):
        """ Crea un ambiente context manager per il server IMAP. Alla chiusura,
        il server IMAP viene notificato della fine della trasmissione con il
        programma. """
        self._imap = imaplib.IMAP4_SSL(
            IMAPAccesser._imap_server_address,
            port=IMAPAccesser._imap_port
        )
        try:
            self._imap.login(self._address, PASSWORD)
            status, _ = self._imap.select('INBOX')
            if 'OK' not in status:
                raise ValueError(
                    "Fallita selezione della casella postale. Per favore, "
                    "verifica che il server IMAP postale sia configurato "
                    "correttamente."
                )
        except imaplib.IMAP4.error:
            raise ValueError(
                "L'API ARPA è stata configurata in maniera incorretta. "
                "Per favore, provvedi a impostare una repository email che"
                "supporti il protocollo IMAP correttamente. (Login fallito, "
                "è possibile che la password fosse errata )"
            )
        self._authenticated = True
        return self

    def _read_available_mails(self) -> tuple[list[int], list[email.message.Message]]:
        """ Leggi le mail non ancora processate dalla casella postale e identifica
        tutte le mail con un id di richiesta compatibile con quelli di cui il
        manager deve occuparsi. """
        if not self._imap or not self._authenticated:
            raise RuntimeError(
                "Non è possibile leggere le mail di un server IMAP non inizializzato. "
                "Se la funzione è stata chiamata indipendentemente dal manager, verifica "
                "che sia avvenuto il login e che il lock sia riservato."
            )
        if not self._available_id_lock.locked():
            raise RuntimeError(
                "Non è possibile chiamare la funzione {func} senza aver prima "
                "ottenuto il lock per gli id richiesti. Per favore, non chiamare "
                "la funzione indipendentemente dal gestore IMAP.".format(
                    func=self._read_available_mails.__name__
                )
            )
        _, unread_emails = self.get_unread_email_indices()
        if len(unread_emails) > 16:
            warnings.warn(
                "La casella postale del server IMAP ha molte mail non lette. Questo può "
                "rallentare il funzionamento dell'API. Per favore, accedi manualmente alle "
                "mail e rimuovi le mail non lette inutili."
            )

        ids, mails = [], []
        for mail_index in unread_emails:
            # Il protocollo IMAP definisce una lista d'indici vuoti l'elemento [b'']
            if mail_index:
                typ, data = self._imap.fetch(mail_index.decode('utf-8'), '(RFC822)')
                packet, flags = data
                protocol_info, raw_email = packet
                mail_string = raw_email.decode('utf-8')
                email_message = email.message_from_string(mail_string)
                if self._verify_sender_identity(email_message):
                    req_id = self._get_response_id(email_message)
                    if req_id and req_id in self._remaining_expected_ids:
                        ids += [req_id]
                        mails += [email_message]

        return ids, mails

    def __call__(self):
        """ Routine che riceve dal server IMAP tutte le email che il gestore si
        aspetta. Blocca l'esecuzione (multithread) finché tutte le email richieste
        non sono ricevute e assegnate ai worker. """

        if not self._imap or not self._authenticated:
            raise RuntimeError(
                "Non è possibile mettere il gestore della API IMAP in ascolto "
                "prima di aver inizializzato il server imap e aver effettuato il "
                "login. Per favore, includi la chiamata al gestore dentro un blocco "
                "di context manager."
            )
        while self._remaining_expected_ids:
            print("Remaining:", self._remaining_expected_ids)
            with self._available_id_lock:
                # È possibile che più di una mail sia arrivata,
                # perciò gotten_ids e mails sono liste.
                gotten_ids, mails = self._read_available_mails()
                print(gotten_ids, mails)
                id_m_pairs = zip(gotten_ids, mails)
                if gotten_ids and mails:
                    for req_id, m in id_m_pairs:
                        if req_id not in self._remaining_expected_ids:
                            raise apierrors.RequestError(
                                "Una mail inaspettata è arrivata al server IMAP associato "
                                "all'API. La mail inaspettata non verrà inclusa nel successivo "
                                "parsing dei CSV. Per favore, verifica che la richiesta sia stata "
                                "effettuata correttamente.",
                                failed_request=f'Richiesta con id {req_id}, mail={m}'
                            )
                    for _id, _mail in id_m_pairs:
                        self._remaining_expected_ids.remove(_id)
                        # Aggiungi una nuova coppia al dizionario e rimuovi l'id recuperato
                        # dalla lista degli id mancanti.
                        self._available_request_ids.update({_id: _mail})
        if self._remaining_expected_ids:
            raise ValueError(
                "Per un errore sconosciuto rimangono degli id presenti dopo "
                "la fine dell'attività del gestore. Verifica che la richiesta "
                "sia stata effettuata correttamente e che il codice sorgente non sia "
                "danneggiato."
            )
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Esci dal context manager. Se un eccezione è avvenuta, segnala un warning all'utente.
        Altrimenti rilascia tutte le risorse e chiudi il server IMAP aperto nel context manager. """
        print("Exited from context manager.")
        if exc_type is not None:
            warnings.warn(
                "Un eccezione è avvenuta durante un context manager del server "
                "IMAP specificato, il server verrà chiuso. Parametri al momento "
                "dell'eccezione: -| Id rimanenti: {ids}, -| Id disponibili {disp} ".format(
                    ids=self._available_request_ids, disp=self._available_request_ids
                )
            )
        # Rilascia tutte le risorse acquisite.
        if self._imap is not None:
            if self._authenticated is True:
                self._imap.logout()
            else:
                self._imap.shutdown()
            self._authenticated = False
        if self._available_id_lock.locked():
            self._available_id_lock.release()
        self._imap = None

    @staticmethod
    def _verify_sender_identity(mail):
        """ Verifica che il mittente della mail sia il servizio di API dell'ARPA. """
        try:
            _sender = mail.get('From')
            _subject = mail.get('Subject')
            _req_tok = 'RICHIESTA DATI RETE METEOIDROGRAFICA ARPA PIEMONTE'
            if _sender != 'virtcsi-iris@arpa.piemonte.it' or _subject != _req_tok:
                raise ValueError
        except (KeyError, ValueError):
            return False
        return True

    @staticmethod
    def _get_response_id(mail):
        """ Ottieni l'ID associato a una risposta API. Se più di un ID appare,
        è probabile sia avvenuto un errore e viene mostrato un warning in uscita.
        Se nella richiesta non è presente alcun ID, viene restituito None. L'ID
        restituito è quello del primo file compatibile trovato.
        Non viene verificato se tutti gli ID di tutti i file sono identici. """

        r_id = re.compile(r'richiesta_dati_gg_(?P<response_id>[0-9]+)')
        for possible_attached in mail.walk():
            f_name, _ = _get_part_file_specs(possible_attached)
            found_id = r_id.findall(f_name)
            if not found_id:  # Prova a parse-are ogni file in cerca dell'id.
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
                # Se più di un id è presente, prendi il primo.
                req_id = int(found_id[0])
            except (ValueError, IndexError):
                raise ValueError("La richiesta api è fallita in quanto la risposta "
                                 "presenta un id scorretto. Per favore, verifica manualmente "
                                 "le mail e correggi l'errore.")

            return req_id
        # Se nessuna parte ha un id valido allora la risposta non ha un id.
        return None


def arpa_data():
    """

    :return:
    """
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
        """ Verifica che il mittente della mail sia il servizio di API dell'ARPA. """
        try:
            _sender = mail.get('From')
            _subject = mail.get('Subject')
            _req_tok = 'RICHIESTA DATI RETE METEOIDROGRAFICA ARPA PIEMONTE'
            if _sender != 'virtcsi-iris@arpa.piemonte.it' or _subject != _req_tok:
                raise ValueError
        except (KeyError, ValueError):
            return False
        return True

    typ, data = imap.search(None, '(UNSEEN)')
    print("Data: ", data)
    typ, data = imap.fetch('4', '(RFC822)')
    # noinspection PyUnresolvedReferences
    em, flags = data
    protocol_info, mail = em
    print("SEPARATE: ", mail)
    print("FLAGWSS", flags)

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

    # get_csv_from_imap_mail()
    for part in email_message.walk():
        # print(part)
        # Cast to string to avoid None
        file_name = str(part.get_filename())
        file_extension = re.compile(r'.*\.(?P<extension>.+)').findall(file_name)
        if not file_extension or 'csv' not in file_extension:
            # Do not attempt parsing
            continue

        main_type = part.get_content_maintype()
        if main_type == 'application':
            if _get_content_transfer_encoding(email_part=part) != 'base64':
                raise ValueError(
                    "Errore interno dell'API: Il file CSV in allegato in una delle risposte è "
                    "stato codificato in un formato diverso da Base64. Perfavore, verifica le "
                    "mail manualmente per verificare che non ci siano errori e in caso aggiungi al "
                    f"codice sorgente un branch per l'encoding {_get_content_transfer_encoding(part)}"
                )

            raw_csv_file = base64.b64decode(str(part.get_payload()))
            csv_file = raw_csv_file.decode('utf-8')
            # imap.store('2', '+FLAGS', '\\Seen')
        elif main_type == 'text':
            csv_file = part.get_payload()
        else:
            # TODO: Add Id
            raise ValueError(
                "Il CSV allegato nella risposta API è in un formato "
                "non riconosciuto dal parser. Perfavore, verifica manualmente "
                "la richiesta con id ID dalle mail e risolvi eventuali "
                "conflitti. Il formato della risposta è {m_ty}".format(m_ty=main_type)
            )

    table_name = 'richiesta_dati.dati_giornalieri'
    req_url = 'https://www.arpa.piemonte.it/radar/open-scripts/richiesta_dati_gg.php?richiesta=1'

    def _issue_arpa_api_request(data, table: str, address: str):

        data = '{"email":"ggriverflow1@outlook.it","data_inizio":"2023-01-24","data_fine":"2023-04-26","richiedente":"4","tipofile":"csv","parametri":["IU"],"stazioni":["379"]}',
        req_data = {
            'data': data,
            'dest_table': table
        }

        # response = requests.post(address, data=req_data)

        def _is_error_response(status_code):
            """ Verifica che la risposta non sia un errore del protocollo HTTPS. """
            first_err_code = 400
            return status_code >= first_err_code

        class response:
            status_code = 200
            text = 'Errore 42coglione'

        if _is_error_response(response.status_code):
            raise requests.HTTPError(
                "Errore nella richiesta HTTP alla API. Questo può verificarsi "
                "quando l'URL di richiesta non è più accettato o per un errore "
                "nei dati richiesti. Verifica che l'URL sia ancora corretto e che "
                "il servizio ARPA sia ancora online.")
        elif 'Errore' in response.text:  # Errore dell'API

            def _is_sql_error(source):
                return re.compile(
                    r'WHERE|ADD|AND|ANY|IF|CREATE|DATABASE|DROP|TABLE|UNION|UPDATE'
                ).findall(source)

            duplicate_keys_error = re.compile(r' +duplicate key value +')
            if duplicate_keys_error.findall(response.text):
                raise apierrors.RequestError(
                    "La richiesta era già presente nel database interno di ARPA, impossibile "
                    "richiedere ancora gli stessi dati. Perfavore, cambia la richiesta "
                    "associata al campo dati: {req_data}".format(req_data=req_data, id=None),
                    failed_request=req_data
                )
            elif _is_sql_error(response.text):
                raise apierrors.RequestError(
                    "Errore interno dell'API: la richiesta inviata con id={id} ha generato un errore "
                    "del database ARPA. Perfavore, verifica la correttezza del codice sorgente o "
                    "verifica di aver inviato correttamente la richiesta. ".format(id=None),
                    failed_request=req_data
                )
        try:
            request_id = int(response.text)
        except ValueError:
            raise apierrors.RequestError(
                "Un errore sconosciuto è emerso durante la richiesta API. "
                "Controlla che la richiesta sia stata effettuata correttamente "
                "e che il server ARPA sia online.",
                failed_request=req_data
            )

        return request_id, response.status_code

    _issue_arpa_api_request('Cazzo', table_name, 'https://www.google.com')


if __name__ == '__main__':
    # arpa_data()


    with IMAPAccesser('ggriverflow1@outlook.it', [15, 23, 56]) as imap:
        pass



