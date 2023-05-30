# Documentazione dell'API

L'api contenuta in /src/api/api.py ha lo scopo di collezionare e raggruppare varie risposte API Arpa in un unico documento.
E' possibile scegliere liberamente sia i dati da includere nel risultato finale sia la loro disposizione e formatting. 

## Lo script di richiesta

La richiesta avviene in seguito alla scrittura e all'esecuzione di uno script di richiesta. 

Lo script di richiesta (SR) è un file testuale col formato 'io' che contiene indicazioni sui dati richiesti e sulle modalità della richiesta. Diverse sezioni permettono di specificare informazioni differenti.

Lo script di richiesta (SR) può essere diviso in macro-aree a discrezione di chi lo scrive. Ogni macroarea è annunciata da uno statement del tipo
```
#<NomeArea>
```
Il nome dell'area è puramente arbitrario, utile solo a chi scrive lo script per orientarsi. Seguono alla dichiarazione della macro area tutte le specifiche riguardanti i dati ad essa associati.

### Le sezioni

Tutte le keyword nello SR sono del tipo `$<Keyword>$`. Dopo ogni keyword deve seguire una newline (eventualmente, cr-lf) e una serie di dichiarazioni.

### La keyword `$Request$`

La keyword di `$Request$` contiene i dati per una singola richiesta ARPA. Un'area può contenere più richieste, che vengono infine raggruppate e salvate in un unico csv. E' possibile specificare la struttura finale del csv nella sezione `$Meta$`, descritta più avanti.



* `stazione: <NomeStazione>`: Il nome della stazione Arpa dalla quale prendere i dati. 
* `dati: <ListaDati>`: la destinazione in cui verrà salvato il csv finale.
* `intervallo: <Password>`: la password corrispondente al sever IMAP di email.
* `format_strings: <Format_string>`: il formato del file csv completo. La stringa di formato segue le stesse specifiche di quelle descritte nella sezione `$Request$`

Ogni stazione ha una lista dei dati disponibili che è possibile richiedere. E' possibile consultare un record dei dati disponibili per ogni stazione direttamente sulla documentazione API Arpa oppure dentro il file `src/api/api_config/stations.json`. 
In generale, i dati disponibili da Arpa sono 
* 'Altezza idrometrica': 'I',
* 'Precipitazione giornaliera': 'P',
* 'Temperatura aria': 'T',
* 'Gradi giorno': 'K',
* 'Vento': 'V',
* 'Neve': 'N',
* 'Radiazioni': 'R',
* 'Umidità': 'H'

La lettera a destra corrisponde all'encoding presente in `stations.json` per le variabili disponibili in ogni stazione. 

La stringa di formato è una stringa che descrive il formato dei csv richiesti dalle stazioni. Il formato dev'essere necessariamente inserito manualmente. La seguente stringa descrive quanto segue
```
>> test.csv
> luca,10,200,ax/100
> marco,4,200,bx/100

format_str = '{nome},{grado},{stipendio},{livello}/{id}
> Risultato: 
nome: luca, grado: 10, stipendio: 200, livello: ax, id: 100 
```
Sia il campo `format_strings` che il campo `dati` rappresentano liste di elementi. 
L'elemento che separa le varie stringhe di formato è il carattere `@`. La virgola non poteva essere usata, in quanto presente nei csv. E' necessario esprimere una stringa di formato per ogni dato richiesto in `dati` all'interno della lista delle `format_strings`. Il seguente snippet mostra un esempio di sezione di `$Request$`.

```
#Vienna
$Request$
stazione: VIENNA;
dati: [Prec, Hum];
intervallo: 2023-01-02*2023-01-12;
format_strings: [{Data},{Precipitazioni},{ClassePrec}@{Data},{Umidita},{Classe}];

$Meta$
email_address: email@gmail.uk;
filename: /your/path/to/file.csv;
final_form: {Data},{Precipitazioni},{Umidita};
$End$
```

Il risultato finale sarà posto in `/your/path/to/file.csv`:
```
Data,Precipitazioni,Umidita
2023/01/02, 0.0, 0.13 
2023/01/03, 0.1, 0.0 
...
2023/01/12, 0.4, 0.03 

```
Nell'esempio sono presenti alcune sezioni delle quali verrà discusso in seguito, come `$End$` e `$Meta$`.
### La keyword `$End$`

La keyword `$End$` ha solo lo scopo di terminare una macro area. E' necessario aggiungere una keyword `$End$` dopo ogni macroarea, anche se è solo una. Il seguente snippet mostra l'uso della keyword `$End$`:
```
#Vienna
...
$End$
#Parigi
...
$End$
```

### La keyword `$Meta$`

Questa keyword permette di specificare le modalità della richiesta per ogni macro area. La keyword `$Meta$` precede una lista di attributi, come per la keyword di `$Request$`. A cambiare sono gli attributi disponibili, e anche il fatto che *ci può essere una sola keyword `$Meta$` per ogni macro area*. I campi riconosciuti attualmente da `$Meta$` sono:

* `email_address: <Email>`: l'email al quale arriveranno le risposte Arpa.
* `filename: <Path>`: la destinazione in cui verrà salvato il csv finale.
* `password: <Password>`: la password corrispondente al sever IMAP di email.
* `final_form: <Format_string>`: il formato del file csv completo. La stringa di formato segue le stesse specifiche di quelle descritte nella sezione `$Request$`

La stringa di formato in `final_form` descrive il modo in cui le variabili dichiarate nelle richieste precedenti verranno disposte all'interno del csv riassuntivo dell'area. 
Tutte le variabili dichiarate nelle stringhe di formato delle richieste di un'area sono disponibili per essere impiegate nella `final_form`.


### La keyword `$Label$`

La keyword `$Label$` permette di assegnare alla sezione che segue immediatamente dopo un nome simbolico al quale è possibile accedere tramite l'interfaccia Python, per poterla modificare facilmente all'interno di Python.

E' necessario che dopo ogni `$Label$` segua una sezione successiva. Nonostante sia possibile assegnare una targhetta alle sezioni `$End$`, questo è inutile in quanto non contengono campi testuali.  

Nel seguente esempio viene dichiarata una sezione di richiesta `$Request$` al quale viene assegnato il nome di `RichiestaFiume`.

```
$Label$ RichiestaFiume
$Request$
..
..
```
Come descritto in seguito, sarà possibile modificare un file di script direttamente da python grazie alle targhette. Al più una targhetta può essere assegnata per ogni sezione. 

## L'interfaccia Python

All'interno di python, è possibile caricare uno script con l'interfaccia fornita da un oggetto di tipo APIConfigEnv (da Api Configuration Environment).

```
file_path = '/your/file/path'
new_script = APIConfigEnv.get_from_file(file_path)
```

La funzione statica `get_from_file` legge il file di script ritorna un oggetto APIConfigEnv che rappresenta il codice sorgente.

E' possibile modificare direttamente da python uno script ben costruito attraverso l'interfaccia di `get_label_section`.

```
new_script = APIConfigEnv.get_from_file(file_path)
section = new_script.get_label_section('YourSection')
```

`section` è un oggetto di tipo EnvLabel, che rappresenta la sezione con la targhetta (label) indicata nell'esempio come `'YourSection'`.

E' possibile modificare le sezioni a cui viene dato un nome tramite targhetta con l'interfaccia di `EnvLabel`:

```
"""
>> Prima: 
>> dentro /script.io

1] $Label$ Esempio
2] $Request$
3] stazione: VIENNA;
4] [...]
"""
section = new_script.get_label_section('Esempio')
section['stazione'] = 'PARIGI'

```
Senza richiedere esplicitamente che la modifica venga salvata, questa rimane solamente all'interno dell'ambiente Python. Per salvare le modifiche nel file è necessario utilizzare `save`.
```
new_script.save('/script.io')

"""
>> Dopo: 
>> dentro /script.io

1] $Label$ Esempio
2] $Request$
3] stazione: PARIGI;
4] [...]
"""
```
Per eseguire uno script è sufficiente utilizzare la funzione `execute`. Il seguente snippet di codice carica uno script preesistente e lo esegue. 

```
new_script = APIConfigEnv.get_from_file(file_path)
new_script.execute()
```
