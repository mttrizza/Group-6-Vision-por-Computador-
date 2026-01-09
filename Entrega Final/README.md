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
├── inference_classifier.py
│
├── model.p
│
├── test_vision.py
├── utils.py
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
│
└── .DS_Store

```

## Descrizione delle cartelle e dei file principali

### utils.py 

#### Scopo del modulo
Il file utils.py contiene la logica matematica di trasformazione dei dati. La sua funzione principale, get_normalized_landmarks, agisce come un filtro intermedio tra l'estrazione grezza di MediaPipe e l'input del classificatore. L'obiettivo è rendere i dati agnostici rispetto alla posizione e alla distanza della mano, garantendo che il modello impari la forma del gesto e non la sua posizione nello spazio.

#### Funzionamento Tecnico
La funzione riceve in input l'oggetto hand_landmarks di MediaPipe e applica una pipeline di trasformazione in tre fasi:

1. Conversione in Coordinate Relative (Invarianza alla Traslazione)I dati grezzi di MediaPipe sono coordinate assolute (x, y) normalizzate rispetto alle dimensioni dell'immagine (0.0 - 1.0). Se usassimo questi dati direttamente, il modello imparerebbe che una mano nell'angolo in alto a sinistra è diversa da una mano nell'angolo in basso a destra, anche se fanno lo stesso gesto.Per risolvere questo problema, il codice imposta il polso (Landmark 0) come origine (0, 0) del sistema cartesiano locale. Sottrae le coordinate del polso da tutti gli altri punti:
```
P'_{i} = P_{i} - P_{polso}
```
```Python
# Trova le coordinate del polso (punto 0) per usarle come origine
if index == 0:
    base_x, base_y = landmark_point[0], landmark_point[1]

# Sottrai la base a tutti i punti (Traslazione dell'origine)
temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y```
```

2. Flattening (Appiattimento) I dati vengono convertiti da una lista di coppie bidimensionali [[x1, y1], [x2, y2]...] a un singolo vettore monodimensionale [x1, y1, x2, y2...].
```Python
# Appiattisci la lista usando itertools
temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
```
3. Normalizzazione di Scala (Invarianza alla Scala)
La mano potrebbe essere vicina alla telecamera (coordinate grandi) o lontana (coordinate piccole). Per rendere il gesto riconoscibile indipendentemente dalla distanza, i valori vengono normalizzati dividendo tutto per il valore assoluto massimo presente nel vettore. Questo costringe tutti i dati a rimanere in un range compreso tra $-1$ e $1$.
```Python
# Normalizza tra -1 e 1
max_value = max(list(map(abs, temp_landmark_list)))

def normalize_(n):
    return n / max_value if max_value != 0 else 0

temp_landmark_list = list(map(normalize_, temp_landmark_list))
```

### create_database.ipynb
#### Scopo del notebook
Questo script costituisce la fase di Pre-processing e Feature Extraction della pipeline di Computer Vision. L'obiettivo non è semplicemente leggere le immagini, ma trasformare i dati non strutturati (pixel delle immagini raw) in dati strutturati (coordinate geometriche dei landmark della mano), pronti per l'addestramento di un classificatore (es. Random Forest).

Nello specifico, il notebook svolge tre compiti critici:
1) Iterazione: Scansiona il dataset organizzato in directory.
2) Feature Extraction: Utilizza MediaPipe Hands per rilevare lo scheletro della mano in ogni immagine ed estrarre le coordinate (x, y) dei 21 punti chiave.
3) Serializzazione: Salva le liste di feature e le relative etichette (labels) in un formato binario compresso (data.pickle), riducendo drasticamente la dimensione dei dati rispetto alle immagini originali e velocizzando il training.

#### Prerequisiti e Librerie
Per l'esecuzione corretta, la struttura delle directory deve seguire la tassonomia delle classi (es. data/A, data/B, etc.). Le librerie principali sono:
- MediaPipe: Per l'estrazione dei landmark scheletrici (il "cuore" del pre-processing).

- OpenCV (cv2): Per la manipolazione delle immagini (conversione BGR -> RGB).

- Pickle: Per la serializzazione degli oggetti Python.

- Matplotlib (opzionale): Per visualizzare le immagini durante il debug.

#### Analisi della Struttura (Dettaglio Code-Level)
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

## train_classifier.ipynb
#### Scopo del notebook
In questo script avviene la transizione dai dati geometrici (le coordinate dei landmark estratte nel passaggio precedente, create_database.ipynb) alla creazione di un modello decisionale capace di classificare nuovi input in tempo reale.

L'obiettivo è addestrare un algoritmo di Apprendimento Supervisionato affinché impari ad associare specifici pattern di coordinate (feature) alle lettere corrispondenti (label).

#### Librerie Utilizzate
- Scikit-learn (sklearn): La libreria standard de facto per il ML in Python. Utilizzata per la gestione dei dataset, la creazione del modello e il calcolo delle metriche.
- Pickle & NumPy: Per la gestione efficiente dei dati serializzati e delle operazioni matriciali.

#### Analisi del Flusso (Dettaglio Tecnico)
Celle 1 & 2 – Caricamento e Preparazione Dati Il notebook inizia caricando il file dataset.pickle generato nella fase precedente. Le liste Python vengono immediatamente convertite in NumPy Arrays che sono ottimizzati per i calcoli vettoriali richiesti dagli algoritmi di Scikit-learn, offrendo prestazioni superiori rispetto alle liste standard.


Cella 3 – Data Splitting e Addestramento (Il Core) Questa cella esegue tre operazioni critiche per la validità scientifica del progetto:

1)Partitioning (Train/Test Split): Il dataset viene diviso in due sottoinsiemi disgiunti:
   - Training Set (80%): Usato dal modello per imparare le regole.
   - Test Set (20%): Usato per valutare le prestazioni su dati "mai visti prima".
```python
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
```

2) Selezione del Modello: È stato scelto il Random Forest Classifier.
Motivazione: È un metodo "Ensemble" che costruisce una moltitudine di alberi decisionali. È particolarmente adatto per questo progetto perché gestisce bene dataset con molte feature (42 coordinate totali) ed è robusto contro l'overfitting (il rischio di imparare "a memoria" invece di generalizzare).
Valutazione (Accuracy): Dopo l'addestramento (.fit), il modello genera predizioni sul Test Set. L'accuratezza (accuracy_score) ci fornisce una metrica percentuale affidabile sulla capacità del modello di generalizzare.
```python
model = RandomForestClassifier()
model.fit(x_train, y_train)
# Fai una prova sui dati di test per vedere quanto è bravo
y_predict = model.predict(x_test)
# Calcola l'accuratezza
score = accuracy_score(y_predict, y_test)
```
Accuratezza del modello: 99.26%.

Cella 4 – Serializzazione del Modello Una volta verificata un'accuratezza soddisfacente (tipicamente > 95%), il modello addestrato viene salvato nel file model.p. Questo file contiene l'intero oggetto Random Forest (con tutti i suoi alberi decisionali e le soglie matematiche calcolate) e sarà l'unico file necessario per lo script di inferenza in tempo reale (inference_classifier.py).
```python
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
```

## collect_data.py
#### Motivazione e necessità dello script
Durante le fasi preliminari del progetto, è stato tentato l'addestramento utilizzando esclusivamente la fusione di due dataset pubblici preesistenti. Tuttavia, i test iniziali hanno evidenziato due criticità fondamentali:
1)Eterogeneità dei dati: I dataset originali presentavano condizioni di illuminazione, sfondi e angolazioni troppo diverse rispetto all'ambiente operativo reale, portando a una scarsa capacità di generalizzazione del modello (Domain Shift).

2)Incompletezza delle classi: Non è stato possibile reperire un dataset esterno che coprisse perfettamente tutte le classi desiderate

Per risolvere queste problematiche senza dover annotare manualmente migliaia di immagini, è stato sviluppato lo script collect_data.py. Questo tool permette di integrare il dataset esistente con immagini acquisite direttamente dall'ambiente di utilizzo finale, migliorando drasticamente la robustezza del modello.

#### Funzionamento Tecnico
Lo script implementa un sistema di acquisizione on-demand. A differenza di una registrazione video continua, questo approccio permette all'utente di posizionare la mano correttamente e salvare il frame solo quando il gesto è perfetto, garantendo  qualità del dato in ingresso.
Il funzionamento si basa su tre blocchi logici:

1. Setup della Camera (Alta Risoluzione) Viene inizializzata la webcam con una risoluzione HD (1280x720). Utilizzare una risoluzione più alta in questa fase è cruciale per garantire che MediaPipe (nello step successivo) riceva dettagli sufficienti per estrarre i landmark con precisione.
```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```
2. Gestione Dinamica delle Classi (File System) Il codice non richiede di pre-creare le cartelle manualmente. Sfruttando la libreria os, lo script verifica l'input da tastiera e gestisce automaticamente la struttura delle directory. Se l'utente preme il tasto "A", lo script controlla l'esistenza della cartella ./data/raw/A, la crea se necessario, e calcola il nome del file progressivo per evitare sovrascritture.
```python
# Convertiamo il codice tasto in lettera (es. 97 -> 'a' -> 'A')
lettera = chr(key).upper()
 
# Gestione automatica della struttura delle cartelle
folder_path = os.path.join(DATA_DIR, lettera)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
```
3. Acquisizione e Salvataggio (I/O) Al momento della pressione del tasto, il frame corrente viene "congelato" e salvato su disco tramite OpenCV. Questo permette di popolare rapidamente le classi meno rappresentate o di aggiungerne di nuove (come i comandi gestuali personalizzati) in pochi secondi.
```python
# Conta quanti file ci sono già per non sovrascrivere
count = len(os.listdir(folder_path))
          
# Salva l'immagine
file_name = f"aa{count}.jpg"
cv2.imwrite(os.path.join(folder_path, file_name), frame)
```

## inference_classifier.py
#### Infrastruttura e Sottosistemi di Supporto
Questo script non è un semplice esecutore sequenziale, ma un sistema che integra diverse tecnologie asincrone.

1. Gestione delle Dipendenze e Importazioni
Il sistema si basa su uno stack tecnologico ibrido:

Computer Vision: cv2 (OpenCV) per la gestione dei frame e mediapipe per l'estrazione scheletrica.

Machine Learning: pickle e numpy per caricare il modello serializzato e gestire l'algebra lineare.

Interfaccia Utente Avanzata: PIL (Pillow) viene introdotto per superare i limiti di OpenCV nel rendering di font TrueType (necessari per caratteri speciali come 'Ñ' o '¿') e numpy per la manipolazione pixel-by-pixel delle icone trasparenti.

Concorrenza: threading è cruciale per evitare che la sintesi vocale (TTS) blocchi il flusso video (freeze), mantenendo il sistema in real-time.



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




