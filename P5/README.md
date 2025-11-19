Buenos días prof. somos Mattia Rizza y Riccardo Belletti

---
## Este es nuestro proyecto de VC de la semana 5
---

[Link Video](https://alumnosulpgc-my.sharepoint.com/:f:/g/personal/mattia_rizza101_alu_ulpgc_es/Emd1JH8HPjRErAHb01fiYUcBdl0M23imuzaqFW7IxEB9VQ?e=G4z5rS)

El primer ejercicio que hemos hecho ha sido realizar un prototipo de libre elección.
Hemos creado este **“filtro de Instagram”** donde creamos dos rectángulos, uno para ocultar la cara con la palabra **“CENSORED”** y otro debajo de la cara con escrito **“PERSONA BUSCADA”**.

Cómo lo hemos hecho:
obviamente hemos añadido la apertura de la webcam y la lectura del frame, hemos puesto el efecto espejo para poder verme en el ordenador como si me viera en un espejo

```python
frame = cv2.flip(frame, 1)
```

hemos hecho la detección del rostro usando **DeepFace** así:

```python
faces = DeepFace.extract_faces(
    img_path=frame,
    enforce_detection=False,
    detector_backend='opencv'
)
```

luego si se detecta la cara:

```python
if len(faces) > 0:
    current_region = faces[0]['facial_area']
```

hacemos la estructura utilizando el diccionario *‘facial_area’*

luego creamos las barras sobre los ojos

```python
eye_y_start = y + int(h * 0.25)
eye_h = int(h * 0.2)
cv2.rectangle(frame, (x, eye_y_start), (x + w, eye_y_start + eye_h), (0, 0, 0), -1)
```
Para escribir el texto **"CENSORED"** hemos usado una escala proporcionada al tamaño del rostro

```python
font_scale = 0.8 * (w / 300.0)
if font_scale < 0.3:
    font_scale = 0.3
```

luego hemos centrado el texto calculando la anchura y lo hemos dibujado

```python
(text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 2)
text_x = x + (w - text_w) // 2
text_y = eye_y_start + (eye_h + text_h) // 2
cv2.putText(frame, "CENSORED", (text_x, text_y), font, font_scale, (255,255,255), 2)
```


luego para la palabra **"PERSONA BUSCADA"** hemos hecho así:

```python
sign_w = int(w * 1.2)
sign_h = int(h * 0.4)
sign_x = x - int((sign_w - w) / 2)
sign_y = y + h + 20
```
```python
cv2.rectangle(frame, (sign_x, sign_y), (sign_x + sign_w, sign_y + sign_h), (255,255,255), -1)
cv2.rectangle(frame, (sign_x, sign_y), (sign_x + sign_w, sign_y + sign_h), (0,0,0), 3)
```

calculando dónde colocarlo y dibujando el rectángulo blanco y el borde negro y luego escribiendo el texto

```python
cv2.putText(frame, wanted_text, (w_text_x, w_text_y), font, w_font_scale, (0,0,255), 3)
```

*después de haberlo escalado y centrado:*

```python
(w_text_w, _), _ = cv2.getTextSize(wanted_text, font, 1, 4)
w_font_scale = sign_w / (w_text_w + 20)
w_text_x = sign_x + (sign_w - w_text_w) // 2
w_text_y = sign_y + (sign_h + w_text_h) // 2
```

---
## Detección de emociones
---

El objetivo principal de este trabajo es la implementación de un detector de emociones en tiempo real que aplica filtros visuales ('reacciones') en función del estado de ánimo del usuario.

El sistema se ha desarrollado en el notebook VC_Entrega5_emociones.ipynb evitando el uso de la función automática analyze(). En su lugar, se ha replicado la arquitectura de extracción de características y clasificación: se utiliza el modelo FaceNet para generar los embeddings faciales y, a partir de estos vectores numéricos, se entrena un clasificador SVM personalizado. Esto permite un control total sobre el proceso de aprendizaje y predicción.

En lugar de usar un dataset enorme completo, utilizamos una parte del FER-2013, que estructuramos dentro de C:/dataset_VC/train y test. Para entrenar elegimos solo cinco emociones: angry, happy, sad, surprise y neutral, que corresponden a las carpetas 0, 3, 4, 5 y 6. Esto lo explicamos también dentro de la función LoadDataset_Emociones, que aparece en la segunda sección del notebook. Allí programamos todo el proceso de lectura de imágenes, redimensionado a la dimensión exacta que pide FaceNet y cálculo de los embeddings usando DeepFace.represent(). Además incluimos un límite de máximo 500 imágenes por clase porque algunas carpetas tenían demasiadas muestras y queríamos evitar que el entrenamiento tardara una eternidad o que el conjunto quedara demasiado desequilibrado.

La primera sección del notebook empieza cargando las librerías principales y, sobre todo, el modelo FaceNet. Aquí utilizamos la función DeepFace.build_model("Facenet"), que es la base para convertir una imagen en un vector numérico. Por ejemplo, el inicio del código es así:
```python
print("Cargando modelo FaceNet...")
model_name = "Facenet"
model = DeepFace.build_model(model_name)
dim = model.input_shape
print(f"Modelo cargado. Tamaño de entrada esperado: {dim}")
```
Este fragmento fue importante porque dim es la resolución exacta que FaceNet necesita (normalmente 160x160). Luego esta información la reutilizamos en la carga del dataset, porque todas las imágenes deben ser redimensionadas a estas dimensiones antes de generar el embedding.

Después de cargar el modelo, en la segunda parte del cuaderno definimos la función LoadDataset_Emociones, que se encarga de leer las carpetas del dataset FER-2013. En esta entrega decidimos trabajar solo con cinco emociones: angry, happy, sad, surprise y neutral. Por eso en el código nos quedamos solo con las carpetas ‘0’, ‘3’, ‘4’, ‘5’ y ‘6’, algo que especificamos aquí:
```python
# 0=Angry, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
target_folders = ['0', '3', '4', '5', '6']
```
La función se ocupa de recorrer cada carpeta, leer hasta un máximo de 500 imágenes por clase (para evitar desequilibrios) y generar el embedding con DeepFace. Una de las líneas clave es esta, donde conviertimos cada imagen en su vector característico:
```python
embedding_objs = DeepFace.represent(
    img_path = img_resized,
    model_name = model_name, 
    enforce_detection = False
)
img_embedding = embedding_objs[0]["embedding"]
```
Elegímos enforce_detection=False porque algunas imágenes del dataset no tenían una detección perfecta y se preferíò saltar esos errores. Al final la función devuelve X, Y y las etiquetas de clase, que serán importantes más adelante.
La tercera sección del cuaderno es donde cargamos el dataset organizado en C:/dataset_VC/train y C:/dataset_VC/test 
(Puedes encontrar este conjunto de datos,dataset_VC, en el enlace del vídeo al principio del readme). Allí se puede ver exactamente cuándo empieza la carga del training:
```python
print("Iniciando carga del dataset de TRAINING...")
X_train, y_train, classlabels = LoadDataset_Emociones(path_train, dim)
```
y después lo mismo con el conjunto de test. Cuando finalmente tenemos todos los embeddings calculados, pasamos a entrenar el clasificador. Para este proyecto escogimos un SVM con kernel RBF, porque suele funcionar bastante bien cuando los datos no son lineales, como en el caso de las caras. El código del entrenamiento está en esta parte:
```python
model_svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')
model_svm.fit(X_train, y_train)
```
na vez entrenado el modelo, lo evaluamos en el set de test y guardamos los resultados. Después guardamos los ficheros necesarios para el prototipo en vivo:
```python
joblib.dump(model_svm, "mio_modello_emozioni.pkl")
joblib.dump(classlabels, "etichette_emozioni.pkl")
```
Con esta parte dejamos listo el “cerebro” del sistema: FaceNet para generar los embeddings y un SVM entrenado con las cinco emociones.

Una vez terminado el entrenamiento en la tercera parte del cuaderno, lo que hacemos en la cuarta sección es recargar el modelo SVM y las etiquetas guardadas, porque no queremos reentrenar cada vez. El código es muy simple:
```python
model_svm = joblib.load("mio_modello_emozioni.pkl")
classlabels = joblib.load("etichette_emozioni.pkl")
print(f"✅ Modelo cargado. El modelo conoce {len(classlabels)} clases.") 
```
La variable classlabels desempeña un papel crítico en la arquitectura del sistema, garantizando la correspondencia biunívoca entre los índices predichos por el SVM y las etiquetas reales. La ausencia de este mecanismo de persistencia podría derivar en un desajuste de los índices y, consecuentemente, en predicciones erróneas.

Para la interpretación semántica de estos resultados, se implementó una estructura de mapeo que traduce los identificadores originales del dataset (ej. '0', '3', '4') a etiquetas legibles:
```python
MAPA_CARPETAS = {
    '0': 'angry',
    '1': 'disgust',
    '2': 'fear',
    '3': 'happy',
    '4': 'sad',
    '5': 'surprise',
    '6': 'neutral'
}
```
Nota sobre la selección de clases: Cabe destacar que, aunque los diccionarios de configuración (MAPA_CARPETAS y COLORES_REACCION) conservan la estructura completa de 7 emociones del estándar FER-2013, el modelo final se ha optimizado utilizando solo un subconjunto de 5 clases. Durante la fase experimental, se entrenó inicialmente el modelo con el set completo; sin embargo, se observó que ciertas emociones generaban confusión y reducían la precisión global. Se optó por filtrar el entrenamiento a las 5 emociones más distintivas para maximizar la robustez. No obstante, se ha mantenido la definición completa en el código para preservar la consistencia con la indexación original del dataset y prevenir excepciones de tipo KeyError o desbordamientos de índice en caso de interactuar con modelos no filtrados.

Dentro del bucle principal leemos cada frame y hacemos una copia para dibujar encima. Una parte muy importante del prototipo es el cálculo del embedding en tiempo real. Aquí volvemos a usar DeepFace.represent, pero esta vez con detector_backend="opencv", También lo hemos probado con SSD y funciona mejor, pero es más lento.
```python
embedding_obj = DeepFace.represent(
    img_path = frame, 
    model_name = "Facenet",
    enforce_detection = True,
    detector_backend = "opencv"
)
embedding = embedding_obj[0]["embedding"]
```
Una vez tenemos el embedding del frame actual, lo pasamos directamente al modelo SVM usando predict_proba() porque queríamos obtener también la confianza:
```python
probabilidades = model_svm.predict_proba([embedding])[0]
indice_max = np.argmax(probabilidades)
confianza = probabilidades[indice_max]
```
Durante las pruebas notamos que, cuando el modelo no estaba seguro, la emoción cambiaba demasiado rápido (lo típico que parpadea entre happy y neutral). Para evitar eso añadimos una condición:
```python
if confianza < 0.30:
    emocion_real = 'neutral'
else:
    nombre_carpeta = classlabels[indice_max]
    emocion_real = MAPA_CARPETAS.get(nombre_carpeta, 'neutral')

```
Con esta regla el prototipo se vuelve mucho más estable y la emoción neutral funciona como un “fallback” cuando el modelo duda. Para nosotros esta parte fue clave porque hicimos el prototipo mucho más agradable de usar. Después de decidir la emoción, dibuja un rectángulo alrededor del rostro usando la información de la detección que devuelve DeepFace (facial_area). También escribe el texto encima del cuadro:
```python
cv2.rectangle(frame_para_mostrar, (x, y), (x+w, y+h), color, 2)
cv2.putText(frame_para_mostrar, texto_reaccion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```
Para que la emoción no sea solo un texto, añadimos también una especie de “filtro visual” que cubre ligeramente toda la pantalla. Por ejemplo, si la emoción es happy, la pantalla se tiñe de amarillo; si es sad, un poco azul; si es angry, rojo:
```python
overlay = frame_para_mostrar.copy()
cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color_fondo, -1)
frame_para_mostrar = cv2.addWeighted(overlay, 0.2, frame_para_mostrar, 0.8, 0)
```

Para completar la descripción del proyecto, queremos también explicar brevemente cómo está organizada la carpeta y cómo se puede ejecutar el prototipo sin tener que repetir todo el proceso de entrenamiento.

Dentro de la carpeta del proyecto los ficheros principales son tres. El primero, VC_Entrega5.ipynb, es el notebook donde está todo el desarrollo: la carga del modelo FaceNet, la función de lectura del dataset, el entrenamiento del SVM y, al final, el prototipo que funciona con la webcam. Los otros dos ficheros son los que guardan el modelo entrenado y sus etiquetas. El archivo mio_modello_emozioni.pkl contiene el SVM ya entrenado con los embeddings del FER-2013. Es literalmente el “cerebro” del sistema, porque es el que decide qué emoción corresponde al embedding generado por FaceNet. El archivo etichette_emozioni.pkl guarda la lista de etiquetas ordenadas, que es fundamental para que el índice devuelto por el SVM coincida con la emoción correcta.

Para finalizar, es pertinente realizar una evaluación sobre el desempeño del sistema en entornos reales. Durante la fase de pruebas, se observó que la respuesta del modelo presenta variaciones en función de la emoción analizada. En teoría el prototipo debe reconocer cinco clases: angry, happy, sad, surprise y neutral. En la práctica descubrimos que las emociones surprise y angry funcionan bastante bien, se distinguen rápido y con buena confianza. En cambio sad y happy a veces se detecta y a veces no; parece ser emociónes más difícil para el modelo, probablemente por la variabilidad del dataset. Probamos incluso a cambiar parte del dataset, pero el resultado no mejoró. Por lo que pudimos investigar, FER-2013 no siempre tiene ejemplos de “happy” muy consistentes, y además la expresión feliz se confunde fácilmente con neutral y surprise.



