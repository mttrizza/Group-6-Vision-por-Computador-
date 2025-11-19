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


#detección de emociones

Este proyecto nació con la idea de construir un pequeño sistema capaz de detectar emociones humanas a partir de la cámara, utilizando DeepFace pero sin depender de las funciones ya hechas como analyze().
El notebook VC_Entrega5.ipynb contiene todo el proceso completo: desde la carga del modelo FaceNet hasta el entrenamiento final y el prototipo “en vivo” que reacciona con un pequeño filtro visual dependiendo de la emoción detectada.

El foco principal del trabajo que es un detector de emociones que funciona en tiempo real y que crea una “reacción” visual en la imagen según el estado de ánimo. Para conseguirlo seguí la misma línea del ejemplo de DeepFace que vimos en clase, especialmente el del fichero VC_P5_deepface_kfold donde yo mismo genero los embeddings y entreno el clasificador. Por eso, en la primera sección del notebook empieza todo cargando el modelo FaceNet (“Cargando modelo FaceNet…”), que es lo que DeepFace usa internamente para convertir un rostro en un vector numérico. Este vector es lo que después alimenta al modelo SVM.

