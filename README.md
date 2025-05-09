# RiverFlow

Monitoraggio e predizione del livello d'altezza di corsi d'acqua


# Prerequisiti

 - Numpy package for Python, version >= 1.21
 - Tensorflow package for Python
 - Matplotlib.Pyplot package for Python

# Ultimi update

 - Supporto per variabili vettoriali (encoding one-hot, bag of words ecc...)
 - Creazione di descrizione delle variabili in input e output del modello nel file di log
 - Direttiva di split per supportare le divisioni di dati in training, test e validazione
 - Funzioni native aggiunte in fase di .act: stack(a,b,...n), one_hot_encode(categorical_var, ordine) per il supporto di feature vettoriali
 - Aggiunta possibilità di valutare espressioni dentro la dichiarazione di un path in .decl
