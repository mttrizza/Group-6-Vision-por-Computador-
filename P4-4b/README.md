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
Después, con el código Python ejecutamos nuestro modelo utilizando el video propuesto por usted, profe, y funcionaba correctamente con este código que nos proporcionó el creador del sitio web utilizamos:
```python
import os
import sys
import argparse
import glob
import time
import csv
    

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Utils ----------------
def iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    if inter <= 0:
        return 0.0
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

def canonical_classname(name: str) -> str:
    s = name.strip().lower().replace('_', ' ').replace('-', ' ')
    s = ' '.join(s.split())
    # targhe
    plate_aliases = {
        'license plate','licence plate','number plate','plate',
        'vehicle plate','car plate','plate number','licence','license'
    }
    if s in plate_aliases:
        return 'license-plate'
    # motorini (se vuoi unificarli)
    moto_aliases = {'motorcycle','motorbike','moped','scooter'}
    if s in moto_aliases:
        return 'motorcycle'
    # auto
    car_aliases = {'car','automobile','vehicle','auto'}
    if s in car_aliases:
        return 'car'
    # persone
    person_aliases = {'person','pedestrian','people'}
    if s in person_aliases:
        return 'person'
    return name

# --------------- Argparse ---------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', type=float, default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')
parser.add_argument('--iou', type=float, default=0.4,
                    help='IoU threshold for matching tracks (default 0.4).')
parser.add_argument('--max_misses', type=int, default=45,
                    help='Frames to keep a track without matches (default 45).')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
IOU_THRESH = float(args.iou)
MAX_MISSES = int(args.max_misses)

# --------------- Checks / model ---------------
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

if isinstance(labels, dict):
    index_to_name = dict(labels)
    class_names = [index_to_name[k] for k in sorted(index_to_name.keys())]
else:
    index_to_name = {i: n for i, n in enumerate(labels)}
    class_names = list(labels)

# Cumulative unique counts + trackers per class
unique_totals = {canonical_classname(n): 0 for n in class_names}
trackers = {canonical_classname(n): [] for n in class_names}  # each: {'id', 'bbox', 'misses'}
next_ids = {canonical_classname(n): 0 for n in class_names}

# --------------- Source parsing ---------------
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source); ext = ext.lower()
    if ext in img_ext_list: source_type = 'image'
    elif ext in vid_ext_list: source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif img_source.startswith('usb'):
    source_type = 'usb'
    try:
        usb_idx = int(img_source[3:])
    except:
        print('USB index must follow "usb", e.g., "usb0".'); sys.exit(0)
elif img_source.startswith('picamera'):
    source_type = 'picamera'
else:
    print(f'Input {img_source} is invalid.'); sys.exit(0)

resize = False
if user_res:
    resize = True
    try:
        resW, resH = map(int, user_res.split('x'))
    except:
        print('Resolution must be WxH, e.g. 640x480.'); sys.exit(0)

# --------------- Open source ---------------
cap = None
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    for f in sorted(glob.glob(os.path.join(img_source, '*'))):
        _, e = os.path.splitext(f)
        if e.lower() in img_ext_list: imgs_list.append(f)
elif source_type in ['video','usb']:
    if source_type == 'video':
        cap = cv2.VideoCapture(img_source)
    else:
        cap = cv2.VideoCapture(usb_idx, cv2.CAP_AVFOUNDATION)
    if not cap or not cap.isOpened():
        print('ERROR: Unable to open video/camera source.'); sys.exit(0)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    if not user_res:
        print('Please specify --resolution WxH for picamera.'); sys.exit(0)
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# --------------- Recorder ---------------
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video/camera.'); sys.exit(0)
    if not user_res:
        print('Please specify --resolution to record.'); sys.exit(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recorder = cv2.VideoWriter('demo1.mp4', fourcc, 30, (resW, resH))
    if not recorder.isOpened():
        print('ERROR: Failed to open VideoWriter.'); sys.exit(0)

# --------------- Viz state ---------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
avg_frame_rate = 0.0
frame_rate_buffer = []; fps_avg_len = 200
img_count = 0

window_name = 'YOLO detection results'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:
    while True:
        t0 = time.perf_counter()

        # ---- Read frame ----
        if source_type in ['image','folder']:
            if img_count >= len(imgs_list):
                print('All images processed.'); break
            frame = cv2.imread(imgs_list[img_count]); img_count += 1
            if frame is None: continue
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret: print('End of video.'); break
        elif source_type == 'usb':
            ret, frame = cap.read()
            if not ret or frame is None: print('Camera read error.'); break
        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if frame is None: print('Picamera read error.'); break

        if resize: frame = cv2.resize(frame, (resW, resH))

        # ---- Inference ----
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Bucket detections by canonical class name
        dets_by_class = {}  # cname -> list of (xmin,ymin,xmax,ymax, classidx, conf)
        object_count = 0

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = map(int, xyxy.tolist())
            cls_idx = int(detections[i].cls.item())
            raw_name = index_to_name.get(cls_idx, str(cls_idx))
            cname = canonical_classname(raw_name)
            conf = float(detections[i].conf.item())
            if conf < min_thresh: continue

            # draw box
            color = bbox_colors[cls_idx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{cname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            object_count += 1
            dets_by_class.setdefault(cname, []).append((xmin, ymin, xmax, ymax, cls_idx, conf))

        # ---- Tracking & unique counting (for ALL classes, including license-plate) ----
        for cname, dets in dets_by_class.items():
            # ensure dict keys exist (in case cname is only from canonicalization)
            if cname not in trackers:
                trackers[cname] = []
                unique_totals[cname] = 0
                next_ids[cname] = 0

            tracks = trackers[cname]
            nT = len(tracks); nD = len(dets)
            # mark updated tracks
            updated = [False] * nT
            used_tracks = set()
            used_dets = set()

            # Greedy one-to-one matching by IoU (descending)
            # Build all pair (track, det) with IoU >= thresh
            pairs = []
            for ti, tr in enumerate(tracks):
                tb = tr['bbox']
                for di, d in enumerate(dets):
                    db = d[0:4]
                    iouv = iou_xyxy(tb, db)
                    if iouv >= IOU_THRESH:
                        pairs.append((iouv, ti, di))
            # sort by IoU desc
            pairs.sort(reverse=True, key=lambda x: x[0])

            # assign
            for iouv, ti, di in pairs:
                if ti in used_tracks or di in used_dets:
                    continue
                # update track ti with det di
                tracks[ti]['bbox'] = dets[di][0:4]
                tracks[ti]['misses'] = 0
                updated[ti] = True
                used_tracks.add(ti)
                used_dets.add(di)

            # create new tracks for unmatched detections -> NEW UNIQUE objects
            for di, d in enumerate(dets):
                if di in used_dets: continue
                new_id = next_ids[cname]; next_ids[cname] += 1
                tracks.append({'id': new_id, 'bbox': d[0:4], 'misses': 0})
                updated.append(True)
                unique_totals[cname] = unique_totals.get(cname, 0) + 1

            # age & remove stale tracks
            for ti in range(len(tracks)-1, -1, -1):
                if ti >= len(updated) or not updated[ti]:
                    tracks[ti]['misses'] += 1
                if tracks[ti]['misses'] > MAX_MISSES:
                    tracks.pop(ti)

        # increment misses for classes with no detections this frame
        for cname in list(trackers.keys()):
            if cname not in dets_by_class:
                tracks = trackers[cname]
                for ti in range(len(tracks)-1, -1, -1):
                    tracks[ti]['misses'] += 1
                    if tracks[ti]['misses'] > MAX_MISSES:
                        tracks.pop(ti)

        # Overlay 
        if source_type in ['video','usb','picamera']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.putText(frame, f'Number of objects: {object_count}', (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        x2, y2 = 260, 40
        for cname in sorted(unique_totals.keys()):
            tot = unique_totals[cname]
            if tot > 0:
                y2 += 20
                cv2.putText(frame, f'{cname} unique: {tot}', (x2, y2),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,0), 2)

        cv2.imshow(window_name, frame)
        if record: recorder.write(frame)

        # ---- Keys / close ----
        key = cv2.waitKey(0 if source_type in ['image','folder'] else 5)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (27, ord('q'), ord('Q')): break
        elif key in (ord('s'), ord('S')): cv2.waitKey()
        elif key in (ord('p'), ord('P')): cv2.imwrite('capture.png', frame)

        # ---- FPS ----
        t1 = time.perf_counter()
        fps = 1.0 / max(1e-6, (t1 - t0))
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len: frame_rate_buffer.pop(0)
        avg_frame_rate = float(np.mean(frame_rate_buffer)) if frame_rate_buffer else 0.0

finally:
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    print('--- Totali unici per classe (oggetti distinti) ---')
    for cname in sorted(unique_totals.keys()):
        print(f'{cname}: {unique_totals[cname]}')

    #csv
    import csv
    with open("CSV.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "total"])
        for cname in sorted(unique_totals.keys()):
            w.writerow([cname, unique_totals[cname]])
    print("CSV saved in: CSV.csv")

    if source_type in ['video','usb'] and cap is not None:
        cap.release()
    elif source_type == 'picamera' and cap is not None:
        cap.stop()
    if record:
        recorder.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

```

<img src="image/csv.png" width="300" />

y luego lo ejecutamos usando la terminal de Anaconda con esto:
```python
python yolo_detect.py --model "/Volumes/SSD/yolo/my_model/train/weights/best.pt" --source "/Volumes/SSD/yolo/videoProva.mp4" --resolution 1280x720
```

excepto que marca algunas matrículas idénticas como diferentes, porque cuando el coche se acerca, el tamaño cambia y piensa que son matrículas diferentes, intentamos encontrar una solución pero no pudimos mejorar más.

Hemos incluido el vídeo de demostración de la tarea 4 en la entrega del sitio web de ULPGC.

*(Pero para la entrega 4b hemos modificado el archivo .py y también la forma en que lo ejecutamos)*

Ahora pasamos a la parte del 4b.

