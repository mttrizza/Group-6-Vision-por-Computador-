Buongiorno prof siamo Mattia Rizza e Riccardo Belletti e questo è il nostro progetto finale di Vision por Computador

L’obiettivo del progetto è stato applicare in modo pratico i concetti visti a lezione volendo creare un vero e proprio traduttore per la lingua dei segni.
Il progetto non nasce come un lavoro di ricerca avanzata, ma come un esercizio completo e realistico.
Abbiamo preso un'idea che potrebbe sembrare banale inizialmente, ci abbiamo messo del nostro e siamo riusciti a tirare fuori un bel programma che si potrebbe realmente usare per aiutare le persone con questa disabilità.
Lanciato il programma si aprirà il video e l'utente vedrà che ci troviamo in modalità traduttore, in questa modalità l'utente potrà esercitarsi ad imparare la lingua dei segni spagnola  e gli altri simboli che abbiamo aggiunto per implementare delle azioni.
Una volta che l'utente decide di voler scrivere un messaggio potrà fare il segno con le mani per entrare nella modalità scrittura, dentro questa modalità  l'utente potrà comporre  un qualsiasi messaggio lettera per lettera e, una volta concluso, potrà mettere il dito sull'icona del microfono per far pronuciare al computer la frase composta.
Per velocizzare la scrittura abbiamo aggiunto una funzionalità che ti permette di vedere a schermo delle parole consigliate mentre stai scrivendo, e quindi, se per esempio stai scrivendo "pro" verrano fuori alcune parole  come "proyecto" e mettendoci il dito sopra completerà la parola.
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
####Scopo del notebook
Questo script costituisce la fase di Pre-processing e Feature Extraction della pipeline di Computer Vision. L'obiettivo non è semplicemente leggere le immagini, ma trasformare i dati non strutturati (pixel delle immagini raw) in dati strutturati (coordinate geometriche dei landmark della mano), pronti per l'addestramento di un classificatore (es. Random Forest).

Nello specifico, il notebook svolge tre compiti critici:
1) Iterazione: Scansiona il dataset organizzato in directory.
2) Feature Extraction: Utilizza MediaPipe Hands per rilevare lo scheletro della mano in ogni immagine ed estrarre le coordinate (x, y) dei 21 punti chiave.
3) Serializzazione: Salva le liste di feature e le relative etichette (labels) in un formato binario compresso (data.pickle), riducendo drasticamente la dimensione dei dati rispetto alle immagini originali e velocizzando il training.

####Prerequisiti e Librerie
Per l'esecuzione corretta, la struttura delle directory deve seguire la tassonomia delle classi (es. data/A, data/B, etc.). Le librerie principali sono:
- MediaPipe: Per l'estrazione dei landmark scheletrici (il "cuore" del pre-processing).

- OpenCV (cv2): Per la manipolazione delle immagini (conversione BGR -> RGB).

- Pickle: Per la serializzazione degli oggetti Python.

- Matplotlib (opzionale): Per visualizzare le immagini durante il debug.

####Analisi della Struttura (Dettaglio Code-Level)
Cella 1 – Configurazione dell'Ambiente Vengono definiti i percorsi e inizializzato il modello statico di MediaPipe. A differenza dello script in tempo reale, qui configuriamo MediaPipe con static_image_mode=True, ottimizzato per immagini singole ad alta precisione.
```python
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
DATA_DIR = './data'
```
Cella 2 – Estrazione delle Feature (Core Loop)
Questa è la sezione computazionalmente più intensa. Il codice itera su ogni sottocartella (che rappresenta una classe/lettera) e per ogni immagine esegue la conversione.

Passaggi tecnici rilevanti per ogni immagine:

1) Conversione Spazio Colore: MediaPipe richiede input RGB, mentre OpenCV carica in BGR.
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
2) Inferenza MediaPipe: Vengono calcolati i landmark.
```python
results = hands.process(img_rgb)
```
3)Feature Extraction & Normalizzazione (Cruciale):
Se viene rilevata una mano, non ci limitiamo a estrarre le coordinate grezze ($x, y$ rispetto ai bordi dell'immagine). Invece, viene invocata la funzione custom get_normalized_landmarks
```python
if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            normalized_landmarks = get_normalized_landmarks(hand_landmarks)
            data.append(normalized_landmarks)
            labels.append(dir_)
```

Cella 3 – Serializzazione dei Dati I dati processati vengono salvati. Questo passaggio crea un "checkpoint". Se in futuro si vuole cambiare modello di classificazione (es. passare da Random Forest a SVM o Rete Neurale), non sarà necessario riprocessare tutte le immagini, ma basterà caricare questo file pickle.
```python
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
```

#collect_data.py
Lo scopo principale di collect_data.py è:

- acquisire immagini da una sorgente (ad esempio webcam)
- salvare automaticamente le immagini nelle cartelle corrette del dataset
- associare ogni immagine alla classe selezionata

In questo modo è possibile costruire il dataset in modo più ordinato rispetto a una raccolta completamente manuale.

Come funziona

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




