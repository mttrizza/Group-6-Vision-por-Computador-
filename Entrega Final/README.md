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
Trova le coordinate del polso (punto 0) per usarle come origine
```python
if index == 0:
    base_x, base_y = landmark_point[0], landmark_point[1]
```

Sottrai la base a tutti i punti (Traslazione dell'origine)
```python
temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
```


**2. Flattening (aplanamiento)**  
Los datos se convierten de una lista de parejas bidimensionales [[x1, y1], [x2, y2]...] a un único vector unidimensional [x1, y1, x2, y2...].

Appiattisci la lista usando itertools
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

Esta función ejecuta una operación matemática conocida como **Alpha Blending**. En lugar de sobrescribir brutalmente los píxeles del vídeo con los del icono (lo que resultaría en un rectángulo negro alrededor de la imagen), el código calcula una media ponderada para cada píxel.

Analizando el código, vemos primero la separación de canales:

```python
# Separa los canales: BGR (color) y Alpha (transparencia)
overlay_img = overlay_resized[:, :, :3] 
overlay_mask = overlay_resized[:, :, 3:] / 255.0
```

Posteriormente se calcula la máscara inversa para el fondo:

```python
background_mask = 1.0 - overlay_mask
```

Finalmente, ocurre la fusión matricial propiamente dicha:

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

**Gestión asíncrona del audio** (*Multithreading*)  

La interacción **humano-máquina** requiere feedback inmediato. Sin embargo, la librería de síntesis de voz **pyttsx3** opera en modo bloqueante: cuando se ejecuta el comando **engine.say()**, el procesador espera a que la frase termine antes de pasar a la siguiente instrucción.  
En un contexto de vídeo, esto causaría el *“congelamiento”* de la webcam durante varios segundos cada vez que el ordenador habla.

Para mantener el sistema Real-Time, se introdujo la ejecución concurrente mediante el módulo threading. La función run_voice_thread encapsula la lógica de voz en un proceso paralelo:

```python
def run_voice_thread(text):
    t = threading.Thread(target=speak_function, args=(text, VOICE_ID_MANUALE))
    t.start()
```

Al lanzar el hilo con **t.start()**, el sistema operativo crea un nuevo hilo de ejecución para la voz.  
El ciclo principal del vídeo (while True) continúa por tanto girando a 30 FPS sin interrupciones, mientras que en “segundo plano” el motor **TTS (Text-to-Speech)** pronuncia la frase.

---------------------------------------------di di parlare anche della funzione speak_function

**El motor NLP (Natural Language Processing)**

Por último, para soportar la funcionalidad de **“Sugeridor Inteligente”**, se implementó un motor **NLP** ligero basado en diccionario.  
La elección de no utilizar redes neuronales pesadas (como LSTM o Transformers) para esta tarea está dictada por la necesidad de mantener baja la latencia.

El diccionario "**DICCIONARIO**" actúa como una *Knowledge Base* estática. La función get_suggestions_list ejecuta una operación de string-matching optimizada sobre la última palabra parcial introducida:

```python
def get_suggestions_list(current_sentence):
    if not current_sentence: return []
    parts = current_sentence.split(" ")
    last_fragment = parts[-1]
    if len(last_fragment) == 0: return [] 
    matches = []
    for word in DICCIONARIO:
        if word.startswith(last_fragment) and word != last_fragment:
            matches.append(word)
            if len(matches) >= 3: break 
    return matches
```

Este diseño permite obtener sugerencias instantáneas (complejidad computacional mínima) que se actualizan frame a frame mientras el usuario compone el gesto.

El núcleo operativo del script está encapsulado en un **bucle infinito** (while True), que gestiona la sincronización entre la adquisición del mundo real (Webcam) y el renderizado de la información digital **(GUI)**.

**Adquisición y normalización del flujo de vídeo** 

Al inicio de cada iteración, el sistema adquiere el frame bruto de la cámara. Sin embargo, antes de cualquier procesamiento, se ejecutan dos operaciones críticas de pre-processing:

- **Conversión de espacio de color**: *MediaPipe*, al estar entrenado sobre datasets RGB, requiere este formato, mientras que OpenCV adquiere nativamente en BGR.
- **Mirroring (Efecto espejo)**: esta operación es fundamental para la Usabilidad (UX). Sin el volteo horizontal (flip), mover la mano a la derecha provocaría un movimiento a la izquierda en pantalla, creando confusión al usuario.




## Diario
En lo que respecta al **diario** de este *proyecto final*, muy a menudo hemos trabajado de manera presencial, ya fuera después de las clases en la *biblioteca de la universidad* o en otra *biblioteca* cercana a **Las Canteras**.  
Hemos realizado casi todo el proyecto *juntos* y de *forma presencial*, para que ambos pudiéramos entender bien lo que hacía el otro y porque ante cualquier **problema** o **duda**, en *persona* se consigue resolver casi de inmediato, en lugar de hacerlo por *teléfono*.    
Las pocas veces en las que no conseguíamos encontrarnos en persona, utilizábamos **videollamadas por WhatsApp** o, cuando uno podía y el otro no, trabajábamos de **forma individual** enviándonos mensajes cada vez que se realizaba alguna modificación o avance.


## Tecnologie utilizzate

* **Python 3**
* **Jupyter Notebook**
* Librerie standard per la gestione di immagini e file (ad esempio `os`, `opencv`, `numpy`, quando necessario)

Non abbiamo utilizzato framework particolarmente avanzati perché l’obiettivo del progetto era soprattutto capire **il flusso di lavoro**, non ottimizzare le prestazioni.

---

## Metodologia di lavoro
[DA FARE]

## Propuestas de ampliación
- Se podría añadir la posibilidad de **traducir gestos dinámicos** y no solo estáticos, incorporando también la **segunda mano** para la detección.
- Se podría añadir la posibilidad de **modificar el idioma** en el que se quiere hablar y, en consecuencia, cambiar automáticamente el **diccionario** según el idioma seleccionado.
- **Ampliar el diccionario** con muchas más palabras.




