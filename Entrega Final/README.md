# Buenos días profesor, somos Mattia Rizza y Riccardo Belletti y este es nuestro proyecto final de Visión por Computador
<img src="img/TraductorRBMR.png" width="500" />

[Link Projecto y Video y Dataset](https://alumnosulpgc-my.sharepoint.com/:f:/g/personal/mattia_rizza101_alu_ulpgc_es/IgC-0DDnFtsgSYzWowwpVGCvAVrgFpJ8NdjnmPI-oWncQes?e=cIqijs)

El objetivo del proyecto ha sido aplicar de manera práctica los conceptos vistos en clase queriendo crear un verdadero traductor para la **lengua de signos**.  
El proyecto no nace como un trabajo de investigación avanzada, sino como un ejercicio completo y realista. Partimos de una idea que podría parecer banal inicialmente, le hemos añadido nuestro *toque personal* y hemos conseguido sacar un buen programa que realmente se podría usar para ayudar a las personas con esta *discapacidad*.  

Al lanzar el programa se abrirá el **vídeo** y el usuario verá que nos encontramos en modo traductor; en este modo el usuario podrá practicar para **aprender la lengua de signos española** y los otros símbolos que hemos añadido para implementar acciones.  
Una vez que el usuario decide que quiere escribir un mensaje, podrá hacer el **gesto** 🤟 con las manos para entrar en el modo escritura; dentro de este modo el usuario podrá **componer cualquier mensaje letra por letra** y, una vez terminado, podrá poner el dedo sobre el icono del **micrófono** para hacer que el ordenador **pronuncie** la frase compuesta.  

Para **acelerar la escritura** hemos añadido una funcionalidad que te permite ver en pantalla **palabras recomendadas** mientras estás escribiendo, y por lo tanto, si *por ejemplo* estás escribiendo "pro" aparecerán algunas palabras como "proyecto" y poniendo el dedo encima completará la palabra.  

Además de los gestos para escribir las letras, nosotros hemos añadido a nuestro dataset otros **6 gestos** para realizar las siguientes acciones:

- entrar y salir del modo escritura
- borrar toda la frase escrita en modo escritura
- borrar solo la última letra escrita
- añadir un espacio en la frase
- añadir el signo de interrogación de apertura
- añadir el signo de interrogación de cierre
  
A continuación es posible ver la leyenda con todos los gestos utilizables. 


<img src="img/Legenda.jpeg" width="400" />

--- 

## Idea general del proyecto

La idea base ha sido mezclar algunos datasets de imágenes organizados en clases encontrados en **Kaggle** y luego integrarlos con imágenes hechas por *nosotros*, en aquellas letras que nuestro programa tenía *dificultades* para reconocer correctamente.

En particular:

- hemos recopilado imágenes en bruto (raw data) que ya estaban organizadas en carpetas
- hemos escrito scripts de Python para automatizar parte del proceso
- hemos usado un notebook de Jupyter para explorar y verificar el dataset

## Diario y Metodología de trabajo
En lo que respecta al **diario** de este *proyecto final*, muy a menudo hemos trabajado de manera presencial, ya fuera después de las clases en la *biblioteca de la universidad* o en otra *biblioteca* cercana a **Las Canteras**.  
Por la metodologís de trabajo hemos realizado casi todo el proyecto *juntos* y de *forma presencial*, para que ambos pudiéramos entender bien lo que hacía el otro y porque ante cualquier **problema** o **duda**, en *persona* se consigue resolver casi de inmediato, en lugar de hacerlo por *teléfono*.    
Las pocas veces en las que no conseguíamos encontrarnos en persona, utilizábamos **videollamadas por WhatsApp** o, cuando uno podía y el otro no, trabajábamos de **forma individual** enviándonos mensajes cada vez que se realizaba alguna modificación o avance.

Progetto: Traduttore di Lingua dei Segni Spagnola (LSE) basato su Computer Vision
1. Setup dell’Ambiente di Sviluppo
Per garantire riproducibilità e isolamento delle dipendenze, il progetto è stato sviluppato all’interno di un ambiente virtuale Anaconda.

conda create -n progetto_vc python=3.10
conda activate progetto_vc
pip install mediapipe==0.10.9
pip install pyttsx3
pip install pillow
La versione di Python 3.10 è stata scelta per garantire piena compatibilità con MediaPipe e le librerie di supporto.

2. Dataset: Costruzione e Unione delle Fonti
Abbiamo utilizzato e unificato due dataset pubblici scaricati da Kaggle:
* Spanish Sign Language Alphabet Static
* Lenguaje de Signos Español
L’obiettivo dell’unione è stato aumentare la varietà delle mani, delle angolazioni e delle condizioni di illuminazione, migliorando la capacità di generalizzazione del modello.
Durante lo sviluppo ci siamo accorti che alcune lettere (Y, X, W, V, T, H, F) risultavano poco affidabili. Per questo motivo:
* abbiamo raccolto manualmente nuove immagini tramite webcam (script collect_data.py);
* abbiamo sostituito progressivamente parte delle immagini dei dataset originali con dati raccolti da noi, più coerenti con l’ambiente reale di utilizzo.

3. Pre-processing e Standardizzazione dei Dati (utils.py)
Il file utils.py rappresenta il cuore matematico del progetto: funge da “traduttore” tra la visione artificiale e il modello di Machine Learning.
3.1 Obiettivi del Pre-processing
Il pre-processing è stato progettato per garantire:
* Invarianza alla traslazione Il gesto deve essere riconosciuto indipendentemente dalla posizione della mano nell’immagine.
* Invarianza di scala Il gesto deve essere riconosciuto sia con la mano vicina che lontana dalla webcam.
* Compatibilità con modelli di Machine Learning I dati devono essere trasformati in un vettore numerico adatto a un classificatore.
3.2 Pipeline di Elaborazione
La funzione pre_process_landmark applica i seguenti passaggi:
1. Copia di sicurezza Viene creata una deepcopy dei landmark per evitare di modificare i dati usati per il rendering grafico.
2. Relativizzazione delle coordinate
    * Il landmark 0 (polso) viene fissato come origine (0,0).
    * Tutti gli altri punti vengono espressi come differenza rispetto al polso.
3. Flattening La lista di coppie (x, y) viene trasformata in un unico vettore monodimensionale.
4. Normalizzazione Tutti i valori vengono scalati nell’intervallo [-1, 1], migliorando la stabilità numerica e la convergenza del modello.
Output finale: Un vettore di numeri reali pronto per essere fornito al classificatore.

4. Estrazione delle Feature (create_dataset.ipynb)
Questo notebook ha il compito di trasformare le immagini grezze in dati numerici.
Pipeline:
1. Caricamento delle immagini organizzate per classe (A, B, C, …).
2. Rilevamento dei 21 landmark della mano tramite MediaPipe Hands.
3. Applicazione del pre-processing definito in utils.py.
4. Salvataggio dei dati in formato numerico.
Risultato: un dataset strutturato e pronto per l’addestramento.

5. Addestramento del Modello (train_classifier.ipynb)
5.1 Scelte di Progetto
È stato utilizzato un Random Forest Classifier perché:
* è robusto al rumore;
* non richiede feature engineering complesso;
* funziona bene con dataset di dimensioni medio-piccole.
5.2 Fasi di Training
* Suddivisione dei dati:
    * 80% Training Set
    * 20% Test Set
* Addestramento del modello
* Valutazione tramite accuracy score
Se l’accuratezza supera il 95%, il modello viene esportato come file statico:

model.p
Questo file rappresenta il “cervello” dell’applicazione finale.

6. Applicazione in Tempo Reale (inference_classifier.py)
Questo è il file esecutivo, quello che l’utente finale utilizza.
6.1 Funzionalità Principali
* Acquisizione video dalla webcam
* Rilevamento della mano
* Conversione dei dati visivi in dati matematici
* Predizione del segno
* Interfaccia grafica aumentata
6.2 Pipeline Logica
Fase A – Setup
* Caricamento del modello model.p (se presente).
* Modalità fallback se il modello non è disponibile.
Fase B – Detection
* MediaPipe individua i 21 landmark.
* Disegno dello scheletro della mano a schermo.
Fase C – Ponte Visione → AI
* Conversione coordinate normalizzate → pixel.
* Pre-processing tramite utils.py.
Fase D – Inference
* Predizione numerica del modello.
* Traduzione numero → lettera tramite dizionario.
Output visivo:
* Webcam live
* Bounding box della mano
* Lettera riconosciuta

7. Problema Critico: Distinzione tra T e F (Profondità)
Le lettere T e F risultano quasi indistinguibili in 2D (effetto “ombra cinese”).
7.1 Analisi del Problema
* In una webcam 2D le coordinate (x, y) sono quasi identiche.
* Aggiungere immagini al dataset portava a overfitting.
7.2 Soluzione Algoritmica
Abbiamo sfruttato la coordinata Z stimata da MediaPipe:
* Calcolo della differenza di profondità tra:
    * punta dell’indice
    * punta del pollice
Regola:
* indice più vicino alla camera → F
* indice allineato o dietro → T
7.3 Calibrazione Sperimentale
* F: valori fino a -0.036
* T: valori ~ -0.024
* Soglia finale: -0.028
Risultato: distinzione stabile e riproducibile senza riaddestrare il modello.

8. Modalità Scrittura e Gestione dei Comandi
Abbiamo introdotto segni speciali per:
1. Entrare / uscire dalla modalità scrittura
2. Inserire spazi
3. Cancellare tutto
4. Cancellare ultimo carattere
5. Inserire il punto interrogativo
Questo trasforma il riconoscitore in un vero sistema di scrittura gestuale.

9. Text-to-Speech (Accessibilità)
Per rendere il sistema realmente utile a persone con difficoltà vocali, abbiamo integrato la sintesi vocale.
* Libreria: pyttsx3 (offline)
* Voce: spagnola (ricerca automatica nel sistema)
* Attivazione: uscita dalla modalità scrittura
Quando l’utente termina la frase, il sistema legge ad alta voce il testo prodotto.

10. Supporto Unicode (Ñ, ¿)
OpenCV non supporta correttamente caratteri Unicode. Abbiamo quindi integrato Pillow per il rendering del testo:
* Supporto completo a:
    * Ñ
    * ¿
* Font reali (Arial)
* Testo pulito e leggibile

11. Suggeritore Predittivo (NLP Lite)
Abbiamo implementato un sistema di suggerimento lessicale:
* Dizionario interno con ~100 parole spagnole frequenti
* Analisi dell’ultima parola in tempo reale
* Visualizzazione di suggerimenti dinamici
Interazione Touchless
I suggerimenti sono cliccabili senza mouse:
* Hover con l’indice
* Barra di caricamento temporale
* Selezione automatica

12. Interfaccia Grafica: Icona del Microfono
Abbiamo aggiunto un feedback visivo tramite icone PNG con trasparenza:
* mic_blue.png → stato idle
* mic_yellow.png → hover
* mic_green.png → parlato
Se le icone non sono presenti, il sistema usa un fallback grafico, evitando crash.

13. Risultato Finale
Il progetto integra:
* Computer Vision
* Machine Learning
* Sintesi vocale
* NLP
* Interfaccia touchless
Non si tratta solo di “usare una libreria”, ma di progettare un sistema completo, robusto e orientato all’accessibilità.



## Estructura del proyecto

La estructura principal del repositorio es la siguiente:
```
Progetto_VC/
│
├── pycache/
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
│ ├── collect_data.py
│ │
│ ├── raw/
│ │ ├── ABRIR_INTERROGACION/
│ │ ├── BORRAR_LETRA/
│ │ ├── BORRAR_TODO/
│ │ ├── CERRAR_INTERROGACION/
│ │ ├── ESPACIO/
│ │ ├── F/
│ │ ├── H/
│ │ ├── MODO_ESCRITURA/
│ │ ├── S/
│ │ ├── T/
│ │ ├── U/
│ │ ├── V/
│ │ ├── W/
│ │ ├── X/
│ │ └── Y/
│ │
│ └── new_data/
│
│
└── .DS_Store
```


## Descripción de las carpetas y de los archivos principales

### utils.py

**Objetivo del módulo**
El archivo **utils.py** contiene la *lógica matemática de transformación de los datos*. Su función principal, *get_normalized_landmarks*, actúa como un filtro intermedio entre la extracción en bruto de *+MediaPipe** y la entrada del clasificador.   
El *objetivo* es hacer que los datos sean agnósticos respecto a la posición y a la distancia de la mano, garantizando que el modelo aprenda la forma del gesto y no su posición en el espacio.

**Funcionamiento técnico**  
La función recibe como entrada el objeto **hand_landmarks** de MediaPipe y aplica una pipeline de transformación en tres fases:

**1. Conversión a coordenadas relativas (invarianza a la traslación)**  
Los datos en bruto de MediaPipe son coordenadas absolutas (x, y) normalizadas respecto a las dimensiones de la imagen (0.0 - 1.0). Si usáramos estos datos directamente, el modelo aprendería que una mano en la esquina superior izquierda es diferente de una mano en la esquina inferior derecha, aunque hagan el mismo gesto. Para resolver este problema, el código establece la muñeca (Landmark 0) como origen (0, 0) del sistema cartesiano local. Resta las coordenadas de la muñeca a todos los demás puntos:

```python
P'{i} = P{i} - P_{polso}
```
Encuentre las coordenadas de la muñeca (punto 0) para utilizarlas como origen.
```python
if index == 0:
    base_x, base_y = landmark_point[0], landmark_point[1]
```
Resta la base a todos los puntos (traslación del origen)
```python
temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
```


**2. Flattening (aplanamiento)**  
Los datos se convierten de una lista de parejas bidimensionales [[x1, y1], [x2, y2]...] a un único vector unidimensional [x1, y1, x2, y2...].

Aplana la lista utilizando itertools.
```python
temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
```

**3. Normalización de escala (invarianza a la escala)**
La **mano** puede estar cerca de la cámara (coordenadas grandes) o lejos (coordenadas pequeñas). Para hacer que el gesto sea reconocible independientemente de la distancia, los valores se normalizan dividiendo todo por el valor absoluto máximo presente en el vector. Esto fuerza a que todos los datos queden dentro de un rango entre − 1 y 1.

Normalizza tra -1 e 1
```python
max_value = max(list(map(abs, temp_landmark_list)))

def normalize_(n):
    return n / max_value if max_value != 0 else 0

temp_landmark_list = list(map(normalize_, temp_landmark_list))
```

### create_database.ipynb

**Objetivo del notebook**  
Este script constituye la fase de **Pre-processing** y **Feature Extraction** de la pipeline de Computer Vision. El objetivo no es simplemente leer las imágenes, sino transformar los datos no estructurados (píxeles de las imágenes raw) en datos estructurados (coordenadas geométricas de los landmark de la mano), listos para el entrenamiento de un clasificador (por ejemplo Random Forest).

En concreto, el notebook realiza tres tareas críticas:

1. **Iteración**: Escanea el dataset organizado en directorios.  
2. **Feature Extraction**: Utiliza MediaPipe Hands para detectar el esqueleto de la mano en cada imagen y extraer las coordenadas (x, y) de los 21 puntos clave.  
3. **Serialización**: Guarda las listas de features y las etiquetas (labels) en un formato binario comprimido (data.pickle), reduciendo drásticamente el tamaño de los datos respecto a las imágenes originales y acelerando el training.

**Requisitos previos y librerías**
Para la ejecución correcta, la estructura de directorios debe seguir la taxonomía de clases (por ejemplo data/A, data/B, etc.). Las librerías principales son:

- MediaPipe: para la extracción de los landmark esqueléticos (el “corazón” del pre-processing).
- OpenCV (cv2): para la manipulación de imágenes (conversión BGR -> RGB).
- Pickle: para la serialización de objetos Python.
- Matplotlib (opcional): para visualizar las imágenes durante el debug.

**Análisis de la estructura (detalle a nivel de código)**  
Celda 1 – **Configuración del entorno**
Se definen las rutas y se inicializa el modelo estático de MediaPipe. A diferencia del script en tiempo real, aquí configuramos MediaPipe con static_image_mode=True, optimizado para imágenes individuales con alta precisión.

```python
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
DATA_DIR = './data'
```

Celda 2 – **Extracción de las features (Core Loop)**  
Esta es la sección computacionalmente más intensa. El código itera sobre cada subcarpeta (que representa una clase/letra) y para cada imagen ejecuta la conversión.

Pasos técnicos relevantes para cada imagen:

1. **Conversión de espacio de color**: MediaPipe requiere entrada RGB, mientras que OpenCV carga en BGR.
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

2. **Inferencia MediaPipe: se calculan los landmark.**

```python
results = hands.process(img_rgb)
```

3. **Feature Extraction & Normalización** (crucial):
si se detecta una mano, no nos limitamos a extraer coordenadas crudas (x, y respecto a los bordes de la imagen). En su lugar, se invoca la función *custom get_normalized_landmarks*

```python
if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            normalized_landmarks = get_normalized_landmarks(hand_landmarks)
            data.append(normalized_landmarks)
            labels.append(dir_)
```

Celda 3 – **Serialización de los datos**
Los datos procesados se guardan. Este paso crea un “checkpoint”. Si en el futuro se quiere cambiar el modelo de clasificación (por ejemplo pasar de Random Forest a SVM o Red Neuronal), no será necesario reprocesar todas las imágenes, sino que bastará con cargar este archivo pickle.

```python
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
```

### train_classifier.ipynb

**Objetivo del notebook**  
En este script ocurre la **transición** desde los datos geométricos (las coordenadas de los landmark extraídas en el paso anterior, create_database.ipynb) hasta la creación de un modelo de decisión capaz de clasificar nuevas entradas en tiempo real.

El objetivo es entrenar un algoritmo de Aprendizaje Supervisado para que aprenda a asociar patrones específicos de coordenadas (features) con las letras correspondientes (label).

**Librerías utilizadas**  
- *Scikit-learn* (sklearn): librería estándar de facto para ML en Python. Se utiliza para la gestión del dataset, creación del modelo y cálculo de métricas.
- *Pickle & NumPy*: para gestión eficiente de datos serializados y operaciones matriciales.

**Análisis del flujo (detalle técnico)**  
Celdas 1 & 2 – **Carga y preparación de datos**
El notebook comienza cargando el archivo **dataset.pickle** generado en la fase anterior. Las listas Python se convierten inmediatamente en **NumPy Arrays**, optimizados para cálculos vectoriales requeridos por los algoritmos de Scikit-learn, ofreciendo prestaciones superiores respecto a listas estándar.

Celda 3 – **Data Splitting y entrenamiento** (el core)
Esta celda ejecuta tres operaciones críticas para la validez científica del proyecto:

1. **Partitioning** (Train/Test Split): el dataset se divide en dos subconjuntos disjuntos:

**Training Set** *(80%)*: usado por el modelo para aprender las reglas.
**Test Set** *(20%)*: usado para evaluar el rendimiento en datos “nunca vistos antes”.

```python
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
```

2. **Selección del modelo:** se eligió *Random Forest Classifier*.  
*Motivación:* es un método *“Ensemble”* que construye una multitud de árboles de decisión. Es especialmente adecuado para este proyecto porque gestiona bien datasets con muchas features *(42 coordenadas en total)* y es robusto frente al overfitting (el riesgo de aprender “de memoria” en lugar de generalizar).

3. **Evaluación (Accuracy)**: después del entrenamiento (.fit), el modelo genera predicciones sobre el Test Set. La exactitud (accuracy_score) nos proporciona una métrica porcentual fiable sobre la capacidad del modelo para generalizar.

```python
model = RandomForestClassifier()
model.fit(x_train, y_train)
# Haz una prueba con los datos de test para ver qué tan bueno es
y_predict = model.predict(x_test)
# Calcula la accuracy
score = accuracy_score(y_predict, y_test)
```
Exactitud del modelo: 99.26%.

Celda 4 – **Serialización del modelo**
Una vez verificada una exactitud satisfactoria *(típicamente > 95%)*, el **modelo entrenado** se guarda en el archivo **model.p**.  
Este archivo contiene el objeto completo **Random Forest** (con todos sus árboles de decisión y los umbrales matemáticos calculados) y será el único archivo necesario para el script de inferencia en tiempo real (inference_classifier.py).

```python
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
```

### collect_data.py

**Motivación y necesidad del script**

Durante las fases preliminares del proyecto, se intentó el entrenamiento utilizando exclusivamente la fusión de dos datasets públicos preexistentes. Sin embargo, las pruebas iniciales evidenciaron dos criticidades fundamentales:

1. **Heterogeneidad de los datos**: los datasets originales presentaban condiciones de iluminación, fondos y ángulos demasiado diferentes respecto al entorno operativo real, llevando a una baja capacidad de generalización del modelo (Domain Shift).

2. **Incompletitud de las clases:** no fue posible encontrar un dataset externo que cubriera perfectamente todas las clases deseadas.

Para *resolver estas problemáticas* sin tener que anotar manualmente miles de imágenes, se desarrolló el script **collect_data.py**. Esta herramienta permite integrar el dataset existente con imágenes adquiridas directamente en el entorno de uso final, mejorando drásticamente la robustez del modelo.

**Funcionamiento técnico**
El script implementa un sistema de **adquisición on-demand**. A diferencia de una grabación de vídeo continua, este enfoque permite al usuario posicionar la mano correctamente y guardar el *frame* solo cuando el gesto es perfecto, garantizando calidad del dato de entrada.  
El funcionamiento se basa en tres bloques lógicos:

1. **Setup de la cámara** (alta resolución)
Se inicializa la webcam con una resolución **HD (1280x720)**.  
Usar una resolución más alta en esta fase es crucial para garantizar que MediaPipe (en el siguiente paso) reciba suficientes detalles para extraer los landmark con precisión.

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

2. **Gestión dinámica de clases** (File System)
El código no requiere pre-crear las carpetas manualmente. Usando la librería os, el script verifica la entrada del teclado y gestiona automáticamente la estructura de directorios. Si el usuario pulsa la tecla "A", el script comprueba la existencia de la carpeta ./data/raw/A, la crea si es necesario, y calcula el nombre progresivo del archivo para evitar sobrescrituras.

```python
# Convertimos el código de la tecla en letra (ej. 97 -> 'a' -> 'A')
lettera = chr(key).upper()
 
# Gestión automática de la estructura de las carpetas
folder_path = os.path.join(DATA_DIR, lettera)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
```

3. **Adquisición y guardado (I/O)**
En el momento en que se pulsa la tecla, el frame actual se *“congela”* y se guarda en disco mediante **OpenCV**. Esto permite poblar rápidamente las clases menos representadas o añadir nuevas (como los comandos gestuales personalizados) en pocos segundos.

```python
# Cuenta cuántos archivos ya existen para no sobrescribirlos
count = len(os.listdir(folder_path))
          
# Guarda la imagen
file_name = f"aa{count}.jpg"
cv2.imwrite(os.path.join(folder_path, file_name), frame)
```

### inference_classifier.py

**La infraestructura software y los motores de soporte**

El script **inference_classifier.py** no actúa como un simple ejecutor lineal, sino que se configura como un hub de integración que orquesta simultáneamente visión artificial, interfaces gráficas avanzadas, síntesis vocal y lógica predictiva.

Para superar los **límites nativos** de las librerías individuales (como la falta de soporte de transparencia en OpenCV o las operaciones bloqueantes del audio), fue necesario implementar una capa de infraestructura custom antes de entrar en el ciclo principal de procesamiento.

**El motor gráfico avanzado** *(Alpha Blending)*

Uno de los retos al desarrollar interfaces modernas con **OpenCV** es la gestión de la *transparencia*. **OpenCV** gestiona las imágenes como matrices de píxeles *BGR (Blue-Green-Red)* opacos. Para visualizar iconos modernos *(como el micrófono)* con bordes suaves y fondos transparentes, se implementó la función overlay_transparent.

Esta función ejecuta una operación matemática conocida como Alpha Blending. En lugar de sobrescribir directamente los píxeles del vídeo con los del icono (lo que resultaría en un antiestético rectángulo negro alrededor de la imagen), el código calcula una media ponderada para cada píxel basándose en su nivel de transparencia.

Analizando el código, vemos primero la separación y normalización de los canales:

```python
# Separa los canales: BGR (color) y Alpha (transparencia)
overlay_img = overlay_resized[:, :, :3] 
overlay_mask = overlay_resized[:, :, 3:] / 255.0
```

Posteriormente, se calcula la máscara inversa para el fondo (donde el icono es transparente, el fondo debe verse):
```python
background_mask = 1.0 - overlay_mask
```

Finalmente, ocurre la fusión matricial mediante álgebra lineal con NumPy:
```python
# Fusiona las imágenes: (Color del icono * Alpha) + (Fondo * (1 - Alpha))
blended_roi = (overlay_img * overlay_mask + roi * background_mask).astype(np.uint8)
```

Esta única línea de código vectorial permite obtener una interfaz de usuario fluida.

**Renderizado de texto Unicode** (El puente *OpenCV-Pillow*)  

Otra *limitación crítica* de **OpenCV** es la falta de soporte para **conjuntos de caracteres extendidos** (Unicode). Funciones estándar como *cv2.putText* no son capaces de renderizar caracteres como la **Ñ** española o el signo de interrogación invertido **¿**.  
Para resolver el problema, se creó la función **wrapper put_text_utf8**.  
Esta función actúa como un puente entre dos librerías gráficas distintas:

1. Convierte el frame de vídeo del formato OpenCV (array NumPy) al formato Pillow (PIL Image):

```python
img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

2. Utiliza el motor de renderizado de Pillow para dibujar el texto usando una fuente TrueType (arial.ttf), que soporta nativamente todos los glifos internacionales.

3. Reconvierten la imagen procesada al formato BGR de OpenCV para poder mostrarla en vídeo.

Este enfoque híbrido garantiza que la interfaz de usuario sea lingüísticamente correcta sin sacrificar el rendimiento de la pipeline de vídeo.
Además de la conversión de formatos, la función implementa dos mecanismos de seguridad importantes:

1) Fallback de Tipografía (Portabilidad): El sistema intenta cargar la fuente vectorial arial.ttf para asegurar una estética moderna. Sin embargo, dado que las fuentes disponibles varían según el sistema operativo, se encapsuló la carga en un bloque de manejo de errores. Si el archivo no se encuentra, el sistema carga automáticamente una fuente predeterminada en lugar de detener la ejecución:
```python
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    # Mecanismo de seguridad si falta la fuente
    font = ImageFont.load_default()
```
2) Corrección del Espacio de Color: Existe una discrepancia entre cómo las librerías interpretan los colores: OpenCV utiliza el estándar BGR (Blue-Green-Red), mientras que Pillow utiliza RGB. Si pasáramos el color directamente, el texto rojo aparecería azul y viceversa. Por ello, el código realiza una inversión manual de los canales de color antes de dibujar:
```python
# Inversión de canales: de BGR (OpenCV) a RGB (Pillow)
color_rgb = (color[2], color[1], color[0])
```

**Gestión Asíncrona del Audio y Arquitectura Multihilo** (*Multithreading*)  

Uno de los desafíos críticos en los sistemas interactivos en tiempo real es la gestión de la latencia. La operación más costosa en términos de tiempo de ejecución no es el reconocimiento de imagen, sino la síntesis vocal.

La librería pyttsx3 opera nativamente en modo bloqueante: la función engine.runAndWait() detiene la ejecución del procesador hasta que la frase completa ha sido pronunciada. Si el ordenador debe decir "Hola, ¿cómo estás?", el proceso tarda entre 2 y 3 segundos. En una arquitectura de un solo hilo (Single-Threaded), esto implicaría congelar el flujo de vídeo de la webcam durante ese tiempo, destruyendo la experiencia de usuario.

Para resolver este cuello de botella y mantener el sistema fluido a 30 FPS, se implementó una arquitectura Multihilo (Multithreading) que desacopla el bucle de renderizado (Vídeo) del bucle de procesamiento (Audio).
1. Orquestación de Hilos (run_voice_thread)
La función run_voice_thread actúa como el punto de entrada para la ejecución concurrente. En lugar de ejecutar el audio directamente, instancia un Worker Thread:
```python
def run_voice_thread(text):
    t = threading.Thread(target=speak_function, args=(text, VOICE_ID_MANUALE))
    t.start()
```
Desacoplamiento: Al invocar t.start(), el sistema operativo crea un nuevo flujo de ejecución paralelo.

Resultado: El Main Thread (encargado del vídeo y la IA) queda libre inmediatamente para procesar el siguiente frame, mientras que el audio se procesa en segundo plano.

2. Lógica de Configuración Dinámica (speak_function)
La función speak_function, que se ejecuta dentro del hilo secundario, no se limita a reproducir sonido. Implementa una lógica robusta de autconfiguración y localización para garantizar que el sistema funcione correctamente en diferentes ordenadores.

Analizando el código, vemos tres pasos clave:

A. Selección Automática del Idioma: Dado que el proyecto está diseñado para la Lengua de Signos Española, el sistema no asume una configuración predeterminada. En su lugar, itera sobre los drivers de voz instalados en el sistema operativo buscando explícitamente una voz hispana:
```python
voices = engine.getProperty('voices')
for v in voices:
    # Búsqueda heurística de drivers en español
    if "spanish" in v.name.lower() or "esp" in v.name.lower():
        engine.setProperty('voice', v.id)
        break
```
Este algoritmo de búsqueda garantiza la portabilidad del software: funcionará tanto en un Windows configurado en inglés como en uno en español, siempre que exista un paquete de voz compatible.

B. Configuración de Velocidad: Se ajusta la velocidad de habla (rate) a 140 palabras por minuto para asegurar una dicción clara y natural, adecuada para fines educativos o de asistencia.
```python
engine.setProperty('rate', 140)
```
C. Tolerancia a Fallos: Toda la lógica de audio está encapsulada en un bloque try...except.
```python
except Exception as e: print(f"Errore Audio: {e}")
```

**El motor NLP (Natural Language Processing)**

Por último, para elevar la experiencia de usuario con la funcionalidad de “Sugeridor Inteligente”, se implementó un motor de NLP ligero y determinista.

La decisión técnica de no utilizar redes neuronales profundas (como LSTM, Transformers o BERT) para esta tarea específica fue dictada por la necesidad de priorizar la baja latencia. En un sistema de visión artificial que ya consume recursos de GPU/CPU para procesar 30 imágenes por segundo, añadir un modelo de lenguaje pesado habría comprometido la fluidez del vídeo.

Estructura y Algoritmo: El sistema se apoya en una Knowledge Base estática (la lista DICCIONARIO), que ha sido curada manualmente para incluir:
- Palabras de uso común (HOLA, GRACIAS, POR FAVOR).
- Vocabulario específico del contexto académico/universitario (PROYECTO, PROFESOR, EXAMEN, VISIÓN).

La función get_suggestions_list implementa un algoritmo de Búsqueda de Prefijos (Prefix Matching). Analiza la frase en construcción en tiempo real y aísla el último fragmento escrito para ofrecer candidatos compatibles.

```python
def get_suggestions_list(current_sentence):
    if not current_sentence: return []
    parts = current_sentence.split(" ")
    last_fragment = parts[-1] # Aísla el sufijo actual (ej. "PR")
    if len(last_fragment) == 0: return [] 
    matches = []
    for word in DICCIONARIO:
        # Busca palabras que empiecen por el fragmento (startswith)
        # y evita sugerir la palabra si ya está completa
        if word.startswith(last_fragment) and word != last_fragment:
            matches.append(word)
            # Optimización: Early Exit al encontrar 3 candidatos para no saturar la UI
            if len(matches) >= 3: break 
    return matches
```

Este diseño permite obtener sugerencias instantáneas (complejidad computacional mínima) que se actualizan frame a frame mientras el usuario compone el gesto.

**Arquitectura del Ciclo de Ejecución (Runtime Loop)**
Una vez inicializados los subsistemas de soporte (Gráficos, Audio, NLP), el control del programa pasa al núcleo operativo.

El script está encapsulado en un bucle infinito (while True), que actúa como orquestador central gestionando la sincronización estricta entre la adquisición del mundo real (Webcam) y el renderizado de la información digital (GUI).

**Adquisición y normalización del flujo de vídeo** 
Al inicio de cada iteración, el sistema adquiere el frame bruto de la cámara.
```python
while True:
    ret, frame = cap.read()
    if not ret: break
```
Sin embargo, antes de pasar a la fase de inferencia o dibujo, se ejecutan dos operaciones críticas de pre-processing para adecuar los datos:

- Conversión de espacio de color: MediaPipe, al estar entrenado sobre datasets RGB, requiere este formato específico, mientras   que OpenCV adquiere nativamente en BGR. La conversión es necesaria para garantizar la precisión del modelo.
- Mirroring (Efecto espejo): Esta operación es fundamental para la Usabilidad (UX). Sin el volteo horizontal (flip), mover la   mano física hacia la derecha provocaría un movimiento hacia la izquierda en la pantalla (como una cámara de vigilancia),     creando una disonancia cognitiva que haría imposible interactuar con los botones.

#### Rendering dell'Interfaccia Dinamica (GUI)
L'interfaccia utente non è statica, ma contestuale: cambia in base allo stato del sistema. Il codice utilizza una logica condizionale per decidere cosa disegnare.
Modalità Scrittura vs. Attesa: Il booleano is_writing_mode funge da gatekeeper grafico:
- Se False (Attesa): L'interfaccia è minimalista (barra grigia), invitando l'utente a fare il gesto di attivazione ("ROCK").
- Se True (Scrittura): Viene renderizzata la "Dashboard" completa:
  - La Barra Verde in alto, che ospita la frase in costruzione .
  - Il Pulsante Microfono, che non è un'immagine fissa ma un oggetto a stati (Blu = Riposo, Giallo = Hover, Verde = Attivo).
  - I Box dei Suggerimenti, generati dinamicamente iterando sulla lista current_suggestions.

#### Logica dei Pulsanti Virtuali (Touchless Interaction)Uno degli aspetti più innovativi del progetto è l'implementazione di pulsanti cliccabili senza contatto fisico. Poiché non esiste un mouse o un touch screen, il sistema deve simulare il "click" usando solo la posizione della mano.Questo viene realizzato attraverso un algoritmo in tre fasi: Mapping, Collision Detection e Temporal Filtering.A. Mapping delle CoordinateMediaPipe restituisce coordinate normalizzate ($0.0 \rightarrow 1.0$). Per interagire con la GUI, queste devono essere proiettate nello spazio pixel dello schermo ($1280 \times 720$):
```python
index_x = int((1 - hand_landmarks.landmark[8].x) * W)
index_y = int(hand_landmarks.landmark[8].y * H)
```

B. Collision Detection (Rilevamento Collisioni)
Il sistema verifica se il punto $(x, y)$ dell'indice cade all'interno del rettangolo di un pulsante (Bounding Box). Esempio per il tasto "PARLA":
```python
if BTN_PARLA_X < index_x < (BTN_PARLA_X + BTN_PARLA_W) and \
   BTN_PARLA_Y < index_y < (BTN_PARLA_Y + BTN_PARLA_H):
       is_hovering_any_ui = True
```
C. Temporal Filtering (Dwell Time) Il problema principale delle interfacce gestuali è l'effetto "Midas Touch": si rischia di cliccare tutto ciò che si tocca per sbaglio. Per evitare falsi positivi, è stato implementato un meccanismo di Dwell Time (tempo di permanenza). L'utente deve mantenere il dito sul pulsante per un tempo prefissato (es. 1.0 secondo) per confermare l'intenzione.
```python
elapsed = time.time() - hover_start_time
```
E fornisce un Feedback Visivo Progressivo (Barra di caricamento o cambio colore):
```python
# Disegna barra di caricamento gialla proporzionale al tempo trascorso
load_w = int((elapsed / 1.0) * BTN_W)
cv2.rectangle(frame, ..., (BTN_X + load_w, ...), (0, 255, 255), -1)
```
Solo quando elapsed >= 1.0, l'evento viene scatenato (action_triggered_flag = True) e il comando viene eseguito (es. avvio del thread vocale).4. Gestione Dinamica dei SuggerimentiI pulsanti dei suggerimenti non sono fissi. Ad ogni frame, se l'utente sta scrivendo, il sistema ricalcola le coordinate per $N$ pulsanti (dove $N$ è la lunghezza di current_suggestions).
```python
for i, word in enumerate(current_suggestions):
    bx = SUGG_START_X + (SUGG_W + SUGG_GAP) * i
    # ... disegno rettangolo e testo ...
    # ... controllo collisione per ogni i-esimo pulsante ...
```
Questo design permette all'interfaccia di adattarsi: se non ci sono suggerimenti, i pulsanti spariscono; se ce ne sono 3, appaiono ordinatamente affiancati.
Una volta che il sistema ha rilevato che l'utente non sta interagendo con i pulsanti (quindi is_hovering_any_ui == False), entra in gioco la pipeline di riconoscimento gestuale.

Questa fase non si limita a chiedere "che lettera è?", ma applica una serie di filtri logici e temporali per correggere gli errori tipici della visione artificiale.
Il primo passo è l'interrogazione del modello Random Forest. Invece di chiedere semplicemente la classe vincente (model.predict), il codice richiede le probabilità (model.predict_proba).
```python
features = get_normalized_landmarks(hand_landmarks)
prediction_proba = model.predict_proba([np.asarray(features)])
max_prob = np.max(prediction_proba)
```
Questo permette di implementare un Filtro di Confidenza:
```python
if max_prob < MIN_CONFIDENCE:
    # Ignora il gesto se il modello non è abbastanza sicuro
```
Questo impedisce al sistema di scrivere caratteri casuali quando la mano è in transizione o in una posizione ambigua, riducendo drasticamente il "rumore" di fondo.

#### Correzione problemi
I modelli basati solo su immagini 2D spesso confondono gesti simili. Per risolvere questo problema, nel codice sono stati iniettati dei correttori logici basati sulla geometria 3D e sul tempo.

A. Correzione Geometrica 3D (T vs F) Le lettere 'T' e 'F' nella lingua dei segni sono molto simili, ma differiscono nella profondità (quale dito sta davanti). MediaPipe fornisce la coordinata Z (profondità). Il codice calcola la distanza relativa sull'asse Z tra la punta del pollice e quella dell'indice:
```python
diff_z = index_tip_z - thumb_tip_z
if diff_z < SOGLIA_LIMIT: predicted_character = 'F'
else: predicted_character = 'T'
```
B. Analisi Temporale Dinamica (N vs Ñ) La 'N' e la 'Ñ' hanno la stessa forma della mano, ma la 'Ñ' prevede un movimento ondulatorio laterale. Un classificatore statico non può vedere il movimento. Per risolvere ciò, il sistema mantiene una memoria storica (x_history) delle ultime 20 posizioni del polso.
```python
x_history.append(wrist_x)
if len(x_history) > 20: x_history.pop(0)

# Calcola l'ampiezza del movimento recente
movement = max(x_history) - min(x_history)
if predicted_character == 'N' and movement > SOGLIA_MOVIMENTO_N:
    predicted_character = 'Ñ'
```
Se il sistema rileva la forma "N" MA c'è oscillazione significativa, "promuove" la predizione a "Ñ".
#### Stabilizzazione Temporale (Anti-Flickering)
Una volta determinata la lettera (es. "A"), non possiamo scriverla subito. I modelli ML "sfarfallano" (es. A-A-B-A-A) centinaia di volte al secondo. Per evitare di scrivere "AAAAA", è stato implementato un Timer di Conferma (CONFIRMATION_TIME = 1.5 secondi).
Il sistema verifica la stabilità:
```python
is_stable = (predicted_character == last_char_detected)
```
•	Se la lettera cambia, il timer si resetta.
•	Se la lettera rimane la stessa, il timer avanza.
Durante l'attesa, l'utente riceve un feedback visivo immediato: un cerchio di caricamento disegnato attorno alla mano (cv2.ellipse), che si riempie progressivamente. Questo comunica all'utente: "Ho capito che vuoi fare la A, tienila ferma ancora un attimo...".

#### La Macchina a Stati (Esecuzione Comandi)
Quando il timer scade (elapsed >= CONFIRMATION_TIME), il sistema esegue l'azione associata al gesto riconosciuto. Qui il codice agisce come una macchina a stati finiti.
•	Stato 1: Cambio Modalità (SWITCH) Se il gesto è "MODO_ESCRITURA" (Rock), inverte lo stato booleano is_writing_mode.
```python
is_writing_mode = not is_writing_mode
```
•  Stato 2: Editing del Testo Se siamo in modalità scrittura, il gesto viene tradotto in manipolazione della stringa sentence:
•	Caratteri standard: Vengono appesi alla stringa.
•	BACKSPACE: Rimuove l'ultimo carattere (sentence[:-1]).
•	BORRAR_TODO: Rimuove tutto ciò che è stato scritto.
•	SPACE: Aggiunge uno spazio.
Gestione Trigger Unico: La variabile action_just_triggered impedisce che l'azione venga ripetuta all'infinito se l'utente non muove la mano. L'azione avviene una volta sola, poi il sistema attende che il gesto cambi o che la mano si sposti ("Key Up event").
Questa sezione finale analizza come il codice garantisce fluidità e stabilità operativa.
#### Gestione della Concorrenza (Il Problema del TTS)


NE ABBIAMO Già PARLATO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  MA FORSE NE PARLIAMO  MEGLIO ORA
L'operazione più "costosa" in termini di tempo non è il riconoscimento dell'immagine, ma la sintesi vocale. La funzione engine.runAndWait() della libreria pyttsx3 è intrinsecamente bloccante: se il computer deve dire "Buongiorno, come stai?", impiega circa 2-3 secondi. In un'architettura a thread singolo (Single-Threaded), questo significherebbe congelare la webcam per 3 secondi.
Per risolvere questo collo di bottiglia, è stato implementato il Multithreading.
Analizziamo la funzione run_voice_thread:
```python
def run_voice_thread(text):
    t = threading.Thread(target=speak_function, args=(text, VOICE_ID_MANUALE))
    t.start()
```
-  Disaccoppiamento: Quando l'utente preme il pulsante "PARLA", il sistema non esegue l'audio direttamente. Invece, istanzia un oggetto Thread.
 - Esecuzione Parallela: Il metodo .start() ordina al sistema operativo di creare un nuovo flusso di esecuzione (worker thread).
-  Risultato: Il ciclo while True principale (Main Thread) continua immediatamente a processare il frame successivo della webcam senza attendere. L'audio viene riprodotto in parallelo. Questo design pattern è fondamentale nei sistemi Real-Time Interactive, separando il Rendering Loop (video) dal Processing Loop (audio).
#### Robustezza e Gestione degli Errori (Fault Tolerance)
Un software non deve mai, ed è stato blindato contro i fallimenti critici attraverso l'uso strategico dei blocchi try...except.
•	Caricamento Risorse Esterne: All'avvio, lo script tenta di caricare le icone PNG (mic_blue.png, ecc.). Se i file mancano (errore comune quando si sposta il progetto su un altro PC), il codice intercetta l'eccezione e attiva la funzione di fallback create_dummy_icon, generando risorse grafiche procedurali al volo.
```python
except Exception as e:
    print(f"⚠️ Errore caricamento icone: {e}. Uso fallback.")
    icon_blue = create_dummy_icon(...)
```
•	Pipeline di Riconoscimento: Anche durante il ciclo principale, l'elaborazione di MediaPipe o la predizione del modello potrebbero generare errori imprevisti (es. valori NaN, divisioni per zero in casi limite). L'intero blocco logico è protetto:
```python
try:
    features = get_normalized_landmarks(hand_landmarks)
    # ... logica di predizione ...
except Exception as e:
    display_text = "Err"
    # Il programma continua a girare invece di chiudersi
```
Questo garantisce che un singolo frame corrotto non termini l'applicazione.







## Tecnologie utilizzate

* **Python 3**
* **Jupyter Notebook**
* Librerie standard per la gestione di immagini e file (ad esempio `os`, `opencv`, `numpy`, quando necessario)

Non abbiamo utilizzato framework particolarmente avanzati perché l’obiettivo del progetto era soprattutto capire **il flusso di lavoro**, non ottimizzare le prestazioni.

---

## Propuestas de ampliación
- Se podría añadir la posibilidad de **traducir gestos dinámicos** y no solo estáticos, incorporando también la **segunda mano** para la detección.
- Se podría añadir la posibilidad de **modificar el idioma** en el que se quiere hablar y, en consecuencia, cambiar automáticamente el **diccionario** según el idioma seleccionado.
- **Ampliar el diccionario** con muchas más palabras.




