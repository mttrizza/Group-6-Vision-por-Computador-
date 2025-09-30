Buenos días, este es el ejercicio de Mattia Rizza y Riccardo Belletti (GRUPO 6). 
Para realizar estos ejercicios nos hemos confrontado mucho a distancia, intercambiando fotos, videos e incluso el código; ambos hemos trabajado en todos los ejercicios para aprender a entender bien cómo funcionaban. 

Para el ejercicio 1 Hemos usado la imagen del mandril y la hemos primero convertida en escala de grises. Luego hemos aplicado el detector de bordes Canny, que resalta en blanco (valor 255) las zonas donde la imagen cambia de intensidad. 
En ese punto hemos visto el resultado de Canny con bordes blancos sobre fondo negro, Luego hemos sumado, fila por fila, cuántos píxeles eran blancos. Esto nos ha permitido obtener un perfil que muestra cuánta “actividad de bordes” había a lo largo de la altura de la imagen. 
Finalmente hemos representado el resultado con dos gráficos: a la izquierda la imagen clásica de Canny; a la derecha un gráfico con la evolución del número relativo de píxeles blancos por fila. 

Para el ejercicio 2 en cambio hemos empezado cargando la imagen del mandril y la hemos preparada en escala de grises. 
Antes de buscar los bordes hemos aplicado un filtro gaussiano para reducir el ruido. Luego hemos calculado los gradientes con el operador Sobel. Con la combinación de los gradientes horizontal y vertical hemos obtenido la magnitud, es decir, la fuerza del borde. Hemos binarizado la magnitud con el método de Otsu, que separa automáticamente las zonas de borde del resto. 
En paralelo hemos usado también el detector de Canny. Una vez obtenidas las imágenes binarias de los bordes, hemos contado cuántos píxeles de borde hay en cada fila y en cada columna y hemos identificado las zonas donde los bordes están más concentrados. 
Para visualizar mejor el resultado, hemos superpuesto sobre la imagen original unas líneas rojas y verdes para destacar las filas y las columnas con más bordes.

Para el ejercicio 3 el programa abre la cámara del portátil y muestra el video en directo. 
Con las teclas numéricas de 0 a 4 se puede cambiar el filtro aplicado: 
0: imagen original 
1: escala de grises 
2: filtro de estilo pop-art, en el que hemos modificado los canales de color para hacerlo divertido, tomando inspiración del entrega 1 
3: detección de bordes con Canny 
4: binarización simple mediante un umbral. 
En la parte superior de la ventana hemos puesto un texto con el nombre del filtro y hemos indicado los comandos a pulsar en el teclado. También hemos programado el programa de modo que se pudiera salir tanto pulsando q como ESC. 

Y finalmente para el último ejercicio nos hemos inspirado en el video “My little piece of privacy” que hemos visto en clase y hemos intentado realizar algo similar.
Hemos utilizado la técnica de la sustracción de fondo (en nuestro caso con el algoritmo KNN de OpenCV). El programa identifica qué partes de la escena están cambiando y genera una máscara. Para limpiar la máscara hemos aplicado una operación llamada cierre, que elimina pequeños agujeros e imperfecciones. Sobre la máscara hemos detectado los bordes con Canny y luego encontrado los contornos. Hemos dibujado todos los contornos más grandes en verde. 
Hemos calculado su rectángulo delimitador y dibujado sobre la imagen tres líneas verticales: una a la izquierda del rectángulo, una en el centro y una a la derecha.
