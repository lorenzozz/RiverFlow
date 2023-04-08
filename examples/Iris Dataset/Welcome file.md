# Dataset delle iris

Il seguente esempio fa uso della repository di dataset per il machine learning https://archive.ics.uci.edu/ml/index.php.

Il dataset in esame contiene quattro parametri, lunghezza e larghezza del petalo e dello sepalo (polline), che distinguono tre categorie di fiori, le 'iris-versicolor', le 'iris-virginica', e le 'iris-setosa'.

Le due direttive di makefile leggono il dataset dalla memoria, lo riordinano randomicamente mantenendo l'ordine degli oggetti di ogni riga!) e scalano numericamente i vari campi per poi porli in un unico piano come coppia (input-output atteso). 
L'input sarà quindi un vettore di 4 elementi contenente i parametri elencati precedentemente e l'output sarà un numero che codifica uno dei tre tipi di fiori: 0 per le 'iris setosa', 1 per le 'Iris-versicolor' e 2 per le 'Iris-virginica'.

Dopo la compilazione del piano, i dati sono inseriti nel file di salvataggio nel formato .npz di numpy. Saranno poi accessibili tramite la direttiva numpy.load['x'] e numpy.load['y'] o tramite un oggetto DatasetLoader.

Riportiamo un estratto dei file di risultato, dopo il trattamento specificato in precedenza.

	
> \>\> Your/Plan/File
> 
> 'x':
> sepal_length ---  sepal_width--- petal_length --- petal_length
> [  [-0.85273973,  0.14863014, -2.58219178, -1.00890411],
>     [ 0.44726027, -0.25136986,  1.31780822,  0.29109589]
>    [....] the other 144 entries... ]
>    
>    'y':
>    [ [0], # iris_setosa, [2], # iris_virginica, [2] # iris_virginica
 [...] other 144 entries... ]
> 
Il modello ha prodotto i seguenti log:
> \>\>/File/Di/Log
> 
> Changing field error to full_recovery
> Generated 146 data points from input configuration
> Non target variables:['SLength', 'SWidth', 'PLength', 'PWidth'], Target variables: Categories
> Saved model as a x/y pair inside "Your/Plan/File"
> 


