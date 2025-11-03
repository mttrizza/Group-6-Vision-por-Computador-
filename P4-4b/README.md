## Buenos días, profe. Somos Mattia Rizza y Riccardo Beletti, y este es nuestro README del 4 y 4b.

Para la detección inicial de matrículas, automóviles, motos y personas, buscamos información avanzada en Internet y encontramos estos pasos que nos parecieron los mejores para completar nuestra entrega inicial (la entrega 4).
---

Antes que nada, descargamos de Internet algunos **datasets** de coches, motos y personas, logrando obtener *400* imágenes mezcladas.
Las colocamos dentro de una carpeta y las usamos de la siguiente manera: 

Primero abrimos el terminal de Anaconda y escribimos

```python
conda create --name yolo-env1 python =3.12
```

Entramos en nuestro “yolo-env1” e instalamos **label-studio**,
el programa que utilizamos para entrenar YOLO, y luego con


```python
“label-studio start”
```
Lo ejecutamos y lo abrimos actuando como **host local**, interactuando a través de nuestro *navegador*.
Creamos nuestro proyecto, insertamos todas las imágenes y luego, para cada foto, tuvimos que especificar qué era un coche, una moto, una matrícula o una persona.
El procedimiento fue largo, pero cuando lo terminamos fue muy satisfactorio haberlo logrado. 
Lo exportamos en formato **YOLO with image**, y nos generó una carpeta zip que contenía

1. una carpeta con todas las imágenes utilizadas,
2. una carpeta con las “labels”,
3. un archivo con las clases (coche, matrícula, persona, moto),
4. un archivo que es nuestro dataset .json.

Tomamos el archivo zip y lo renombramos **Data**, y lo colocamos dentro de la carpeta donde está la carpeta con todas las imágenes.

Después de tener estas dos carpetas, pasamos a la parte donde empezamos a entrenar nuestro modelo utilizando *Google Collabs*, que es un servicio donde podemos escribir y ejecutar archivos Python en un navegador web utilizando la GPU ofrecida.
[Google Collabs](https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb)

Primero nos conectamos al sitio, luego, ejecutando este código

```python
!nvidia-smi
```

verificamos que la GPU estuviera activa,
subimos nuestra carpeta *Data.zip*, la descomprimimos y le cambiamos el nombre a *custom_data* con este código


``` python
!unzip -q /content/data.zip -d /content/custom_data
```

Hecho esto, ahora estamos listos para dividir los archivos en carpetas de **Entrenamiento** y **Validación**,
donde la primera carpeta contendrá las imágenes efectivamente utilizadas por el modelo, mientras que la segunda carpeta contendrá las imágenes utilizadas para verificar el rendimiento después de cada entrenamiento. 
Ejecutando este script proporcionado por el creador del sitio
 
``` python
!wget -O /content/train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py
# TO DO: Improve robustness of train_val_split.py script so it can handle nested data folders, etc
!python train_val_split.py --datapath="/content/custom_data" --train_pct=0.9
``` 
creará automáticamente la estructura de carpetas requerida y moverá aleatoriamente el 90% del conjunto de datos a la carpeta “entrenamiento” y el 10% a la carpeta “validación”.

Cuando el script haya terminado, encontraremos una carpeta con las diversas subdivisiones

<img src="image/labels.png" width="300" />

Posteriormente instalamos Ultralytics, que es la librería de Python que utilizaremos para entrenar el modelo YOLO.
Una vez finalizada la instalación, tuvimos que crear un archivo de configuración del entrenamiento; este archivo define la ubicación de las carpetas de Entrenamiento y Validación y establece las clases de los modelos utilizando este código proporcionado por el creador del sitio.

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

Ejecutando este código obtendremos el archivo *Data.yaml*.

Hecho esto, estábamos listos para nuestro **entrenamiento**.
Solo teníamos que decidir qué modelo de YOLO usar, y utilizamos el modelo *YOLO 11s* con este código:


```python
!yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=60 imgsz=640
```

Iniciamos el entrenamiento.

Una vez finalizado, comprimí y descargué el modelo y lo renombré como *my_model.pt*.

Cierro todo, vuelvo a abrir el terminal de Anaconda, entro en **yolo-env1**, luego entro en la carpeta donde coloqué mi archivo *my_model.pt* y también aquí instalo **ultralytics**.
Después, con el código Python ejecutamos nuestro modelo utilizando el video propuesto por usted, profe, y funcionaba correctamente tranne che alcune targhe uguali le segna come diverse, perche quando la macchina si avvicina, cambia la dimensione e pensa siano targhe diverse, abbiamo provato a trovare una soluzione ma non siamo riusciti a migliorare più di così. 


Ahora pasamos a la parte del 4b.

