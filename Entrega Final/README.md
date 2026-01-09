Buongiorno prof siamo Mattia Rizza e Riccardo Belletti e questo è il nostro progetto finale di Vision por Computador

L’obiettivo del progetto è stato applicare in modo pratico i concetti visti a lezione volendo creare un vero e proprio traduttore per la lingua dei segni.
Il progetto non nasce come un lavoro di ricerca avanzata, ma come un esercizio completo e realistico.
Abbiamo preso un'idea che potrebbe sembrare banale inizialmente, ci abbiamo messo del nostro e siamo riusciti a tirare fuori un bel programma che si potrebbe realmente usare per aiutare le persone con questa disabilità.
Una volta lanciato il programma si aprirà il video e l'utente vedrà che ci troviamo in modalità traduttore, in questa modalità l'utente potrà esercitarsi ad imparare la lingua dei segni spagnola  e gli altri simboli che abbiamo aggiunto per implementare delle azioni.
Una volta che l'utente decide di voler scrivere un messaggio potrà fare il segno con le mani per entrare nella modalità scrittura, dentro questa modalità  l'utente potrà comporre  un qualsiasi messaggio lettera per lettera e, una volta concluso, potrà mettere il dito sul pulsante "parla" per far pronuciare al computer la frase composta.
Oltre ai segni per scrivere le lettere, noi abbiamo aggiunto al nostro dataset altri 6 segni per fare le seguenti azioni:
1) entrare e uscire dalla modalità scrittura
2) cancellare tutta la frase scritta in modalità scrittura
3) cancellare solo l'ultima lettera scritta
4) aggiungere uno spazio nella frase
5) aggiungere il punto di domanda di apertura
6) aggiungere il punto di domanda di chiusura

Qui di seguito è possibile vedere la legenda con tutti i segni utilizzabili.
[magari qui mettiamo l'immagine dei vari segni con le corrispettive lettere]

---

## Idea generale del progetto

L’idea di base è stata quello di mixare alcuni **dataset di immagini organizzati in classi** trovati su Kaggle e poi integrati con delle immagini fatte da noi, sulle lettere che il nostro programma faceva fatica a riconoscere correttamente.

In particolare:
- abbiamo raccolto immagini grezze (raw data) che erano già organizzate in cartelle
- abbiamo scritto script Python per automatizzare parte del processo
- abbiamo usato un notebook Jupyter per esplorare e verificare il dataset

---

## Struttura del progetto

La struttura principale del repository è la seguente:

```
Progetto_VC/
│
├── __pycache__/
│
├── create_dataset.ipynb
├── train_classifier.ipynb
│
├── data/
│   ├── collect_data.py
│   │
│   ├── raw/
│   │   ├── ABRIR_INTERROGACION/
│   │   ├── BORRAR_LETRA/
│   │   ├── BORRAR_TODO/
│   │   ├── CERRAR_INTERROGACION/
│   │   ├── ESPACIO/
│   │   ├── F/
│   │   ├── H/
│   │   ├── MODO_ESCRITURA/
│   │   ├── S/
│   │   ├── T/
│   │   ├── U/
│   │   ├── V/
│   │   ├── W/
│   │   ├── X/
│   │   └── Y/
│   │
│   └── new_data/
│
├── inference_classifier.py
│
├── model.p
│
├── test_vision.py
├── utils.py
│
└── .DS_Store

```

### Descrizione delle cartelle e dei file principali

## create_database.ipynb
Scopo del notebook
L’obiettivo di create_dataset.ipynb è:
- leggere tutte le immagini presenti nelle cartelle del dataset
- associare a ogni immagine la corretta etichetta (label)
- salvare i dati in un unico file (dataset.pickle) pronto per essere utilizzato nello step successivo

In pratica, questo notebook rappresenta il ponte tra i dati grezzi (raw) e la fase di training del modello.

Prerequisiti

Prima di eseguire il notebook è necessario che:

-la cartella data/ sia presente nella directory del progetto
- il file utils.py sia nella stessa cartella del notebook
- le immagini siano organizzate in sottocartelle, una per ogni classe

Le principali librerie utilizzate sono:

- os per la gestione dei file
- cv2 (OpenCV) per la lettura delle immagini
- pickle per il salvataggio del dataset

Struttura del notebook

Il notebook è suddiviso in poche celle, ognuna con uno scopo chiaro.

Cella 1 – Importazioni e setup
In questa parte vengono importate le librerie necessarie e viene definita la directory principale dei dati (DATA_DIR).
Qui assumiamo che tutte le immagini siano già state raccolte e organizzate correttamente. Non vengono fatti controlli particolarmente robusti sugli errori, perché abbiamo lavorato su un dataset relativamente piccolo e controllato manualmente.

Cella 2 – Creazione del dataset
Questa è la parte centrale del notebook.
si inizializzano due liste: una per i dati (data) e una per le etichette (labels)
si scorre ogni cartella (classe)

per ogni immagine:

- viene letta con OpenCV
- viene associata la label corretta
- viene aggiunta alle liste
Il processo è completamente automatico, ma si basa sul fatto che la struttura delle cartelle sia corretta. Se una cartella è sbagliata, anche la label lo sarà.

Cella 3 – Salvataggio del dataset

Una volta completata la lettura di tutte le immagini, i dati vengono salvati in un file dataset.pickle usando la libreria pickle.
Questo file viene poi utilizzato negli altri notebook o script per:

- addestrare il classificatore
- testare le prestazioni del modello

#collect_data.py
Lo scopo principale di collect_data.py è:

- acquisire immagini da una sorgente (ad esempio webcam)
- salvare automaticamente le immagini nelle cartelle corrette del dataset
- associare ogni immagine alla classe selezionata

In questo modo è possibile costruire il dataset in modo più ordinato rispetto a una raccolta completamente manuale.

Come funziona a grandi linee

Lo script segue una logica semplice:

- inizializza la sorgente video
- permette di selezionare una classe (o una modalità)
- acquisisce frame dall’ingresso video
- salva i frame come immagini nelle cartelle del dataset

Non è pensato per essere uno strumento robusto o definitivo, ma come un supporto pratico per il progetto.

Librerie utilizzate

Le principali librerie utilizzate sono:

- cv2 (OpenCV) per l’acquisizione delle immagini
- os per la gestione delle directory

Non sono state utilizzate librerie avanzate di data acquisition o interfacce grafiche, per mantenere il codice semplice e leggibile.

Struttura dei dati generati

Le immagini salvate dallo script finiscono nella cartella:

- data/raw/

Ogni classe corrisponde a una sottocartella (lettere, simboli o comandi). Lo script assume che questa struttura esista già o venga creata automaticamente.

Il nome dei file segue una numerazione progressiva, sufficiente per distinguere le immagini all’interno della stessa classe.

Queste semplificazioni sono state accettate per concentrarci sugli aspetti principali del corso, piuttosto che sulla robustezza del software.

Ruolo nel progetto

collect_data.py rappresenta il primo passo della pipeline:

raccolta delle immagini

creazione del dataset (create_dataset.ipynb)

training del classificatore

Senza questo script, la costruzione del dataset sarebbe stata molto più lenta e disordinata.
---

## Tecnologie utilizzate

* **Python 3**
* **Jupyter Notebook**
* Librerie standard per la gestione di immagini e file (ad esempio `os`, `opencv`, `numpy`, quando necessario)

Non abbiamo utilizzato framework particolarmente avanzati perché l’obiettivo del progetto era soprattutto capire **il flusso di lavoro**, non ottimizzare le prestazioni.

---

## Metodologia di lavoro

Il progetto è stato sviluppato in modo iterativo:

1. raccolta manuale delle immagini
2. organizzazione iniziale in cartelle
3. scrittura di script per automatizzare operazioni ripetitive
4. verifica del dataset tramite notebook

Durante il lavoro ci siamo resi conto che:

* la qualità dei dati è fondamentale
* anche piccoli errori di organizzazione rendono difficile l’uso del dataset
* molte decisioni pratiche non erano ovvie senza esperienza

Per questo alcune scelte non sono “perfette”, ma riflettono il nostro livello e il contesto didattico del corso.




