## Buongiorno prof. siamo Mattia Rizza e Riccardo Beletti e questo è il nostro README del 4 e 4b.

Per la rilevazione iniziale delle **targhe, automobili, moto** e persone siamo andati alla ricerca di informazioni avanzate su internet e abbiamo trovato questi passaggi che ci sono sembrati migliori per portare a termine la nostra consegna iniziale (la consegna 4).
---

Prima di tutto abbiamo scaricato da internet qualche **dataset** di macchine, moto e persone riuscendo a ricavare *400* immagini miste,
le abbiamo messe dentro ad una cartella e le abbiamo usate così: 

Per prima cosa abbiamo aperto il terminale di anaconda e scritto 
```python
conda create --name yolo-env1 python =3.12
```
Siamo entrati nel nostro “yolo-env1” e abbiamo installato **label-studio**, 
programma che abbiamo utilizzato per addestrare YOLO e poi con 
```python
“label-studio start”
```
l’abbiamo fatto partire e così l’abbiamo aperto facendo da **host localmente** interagendo attraverso il nostro *browser*,
abbiamo creato il nostro progetto, inserito tutte le immagini e poi per ogni foto abbiamo dovuto specificare quale fosse una macchina, moto, targa e persona. 
Procedimento lungo però quando l’abbiamo finito è stato molto soddisfacente avercela fatta. 
L’abbiamo esportato in **YOLO with image** e ci ha creato una cartella zip con all’interno 
1. una cartella con tutte le immagini utilizzate,
2. una cartella con i “label”,
3. un file con le classi (macchina, targa, persona, moto)
4. file che è il nostro dataset .json.

Abbiamo preso il file zip e l’abbiamo rinominato **Data**, e messo dentro alla cartella dove è presente la cartella con tutte le immagini 

Dopo aver preso queste due cartelle, passiamo alla parte dove iniziamo ad addestrare il nostro modello utilizzando *Google Collabs* che è un servizio dove possiamo scrivere e runnare file python in un web browser utilizzando GPU offerta 
[Google Collabs](https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb)

Per prima cosa ci siamo connessi al sito, poi runnando questo codice 
```python
!nvidia-smi
```
abbiamo verificato che la GPU sia stata attiva, 
abbiamo caricato la nostra cartella *Data.zip* e l’abbiamo unzippata e cambiato nome in *custom_data* con questo codice 
``` python
!unzip -q /content/data.zip -d /content/custom_data
```

Fatto questo ora siamo pronti a dividere i file i cartelle di **Addestramento** e **Validazione**, 
dove la prima cartella conterrà le immagine effettive utilizzate dal modello, mentre la seconda cartella contiene le immagini utilizzate per verificare le prestazioni dopo ogni addestramento. 
Premendo questo script dato dal creatore del sito 
``` python
!wget -O /content/train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py
# TO DO: Improve robustness of train_val_split.py script so it can handle nested data folders, etc
!python train_val_split.py --datapath="/content/custom_data" --train_pct=0.9
``` 
ci creerà automaticamente la struttura di cartelle richiesta e sposterà casualmente il 90% del set di dati nella cartella “addestramento” e il 10% nella cartella “validazione”


Quando lo script ha terminato, troveremo una cartella con le varie suddivisioni 
<img src="image/luz.png" width="300" />

Successivamente abbiamo installato Ultralytics che è la libreria python che utilizzeremo per addestrare il modello YOLO 
Terminata l’installazione abbiamo dovuto creare un file di configurazione dell’addestramento, questo file imposta la posizione delle cartelle di Addestramento e Convalida e definisce le classe dei modelli utilizzando questo codice datosi dal creatore del sito

```python
# Python function to automatically create data.yaml config file
# 1. Reads "classes.txt" file to get list of class names
# 2. Creates data dictionary with correct paths to folders, number of classes, and names of classes
# 3. Writes data in YAML format to data.yaml

import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):

  # Read class.txt to get class names
  if not os.path.exists(path_to_classes_txt):
    print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
    return
  with open(path_to_classes_txt, 'r') as f:
    classes = []
    for line in f.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())
  number_of_classes = len(classes)

  # Create data dictionary
  data = {
      'path': '/content/data',
      'train': 'train/images',
      'val': 'validation/images',
      'nc': number_of_classes,
      'names': classes
  }

  # Write data to YAML file
  with open(path_to_data_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
  print(f'Created config file at {path_to_data_yaml}')

  return

# Define path to classes.txt and run function
path_to_classes_txt = '/content/custom_data/classes.txt'
path_to_data_yaml = '/content/data.yaml'

create_data_yaml(path_to_classes_txt, path_to_data_yaml)

print('\nFile contents:\n')
!cat /content/data.yaml
```

Runnato questo codice avremo il file *Data.yaml*

Fatto questo eravamo pronti per il nostro **addestramento**, 
ora dovevamo solo decidere quale modello di YOLO e abbiamo utilizzato il modello *YOLO 11s* e con questo codice a
```python
!yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=60 imgsz=640
```

Abbiamo fatto partire l’addestramento 

Finito l’addestramento ho compresso e scaricato il modello e l’ha rinominato in *my_model.pt*

Chiudo tutto, riapro il terminale di anaconda, entro in **yolo-env1**, poi entro nella cartella dove ho messo il mio file *my_model.pt* e anche qui installo **ultralytics**, 
poi con il codice python abbiamo fatto partire il nostro modello utilizzando il video proposto da voi prof e funzionava correttamente  

Ora raccontiamo della parte del 4b
