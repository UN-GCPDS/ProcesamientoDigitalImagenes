# %% [markdown]
# ## 1. Definición Formal del Problema
# 
# La detección de objetos se enfoca en una tarea más específica y exigente que la clasificación de imágenes o la segmentación semántica. En lugar de asignar una etiqueta global o clasificar píxeles sueltos, el objetivo es **localizar todas las instancias de una categoría de objeto** mediante cuadros delimitadores (*bounding boxes*).
# 
# En pocas palabras, el modelo no solo responde a la pregunta "¿está el objeto en la imagen?", sino que también debe resolver obligatoriamente "¿dónde está?".
# 
# ### 1.1. Formulación Matemática
# 
# Formalmente, la función de nuestro modelo de detección (generalmente una red neuronal profunda parametrizada por los pesos $\theta$) se define como:
# 
# $$f_\theta: \mathbb{R}^{H \times W \times 3} \rightarrow \mathcal{B}$$
# 
# Donde:
# * $H, W$: Son el alto y ancho de la imagen de entrada a color (3 canales).
# * $\mathcal{B} = \{ (b_i, s_i) \}_{i=1}^N$: Es el conjunto de $N$ detecciones predichas por la red.
# * $b_i = (x, y, w, h)$: Es el cuadro delimitador que encierra al objeto, definido por sus coordenadas, ancho y alto.
# * $s_i$: Es la puntuación de confianza (*confidence score*) que indica la certeza del modelo de que la caja $b_i$ realmente contiene un objeto.
# 
# ### 1.2. Desafíos Principales en la Detección
# 
# Para lograr que el modelo pase de los píxeles de entrada a estas cajas delimitadoras, la arquitectura y el entrenamiento deben superar tres obstáculos fundamentales:
# 
# * **Variación de Escala:** Un mismo objeto (como un vehículo o una persona) puede abarcar desde unos pocos píxeles hasta ocupar casi todo el encuadre. El modelo debe ser capaz de "ver" y procesar características a múltiples resoluciones.
# * **Oclusión:** En entornos reales, los objetos suelen estar parcialmente tapados. La red debe aprender a inferir la caja delimitadora completa $b_i$ apoyándose en partes visibles limitadas.
# * **Desbalance Extremo de Clases (Fondo vs. Objeto):** Si analizamos todas las posibles ubicaciones en una imagen, la inmensa mayoría corresponden simplemente a "fondo" sin interés. Este desbalance causa que, durante el entrenamiento, los aciertos en el fondo abrumen a los errores en los objetos reales, requiriendo el uso de funciones de pérdida especializadas \cite{lin2017focal}.

# %% [markdown]
# ## 2. Arquitecturas Representativas en Detección de Objetos
# 
# Históricamente, los modelos de detección basados en *Deep Learning* se han dividido en dos grandes familias: los detectores de dos etapas (*two-stage*) y los de una etapa (*one-stage*). Cada familia representa un compromiso diferente entre la precisión espacial y el costo computacional.
# 
# ### 2.1. Faster R-CNN (Two-Stage Detector)
# Propuesto por Ren et al. \cite{ren2015faster}, es el arquetipo de los detectores de dos etapas y se ha mantenido como el estándar de oro para aplicaciones que priorizan la precisión. Funciona en dos fases claramente diferenciadas:
# 
# 1. **Region Proposal Network (RPN):** Una sub-red convolucional ligera que se desliza sobre los mapas de características. Su función exclusiva es proponer "Regiones de Interés" (RoIs): cajas delimitadoras candidatas que tienen una alta probabilidad de contener *algún* objeto, discriminando únicamente entre *foreground* y *background*.
# 2. **Detector Head:** Para cada RoI propuesta, se extrae un parche de características de tamaño fijo (mediante *RoI Align*). Posteriormente, una red densa clasifica la región específica y aplica una regresión final para afinar las coordenadas de la caja.
# 
# **Innovación clave:** La RPN permitió que la red aprendiera a proponer regiones internamente, posibilitando el entrenamiento *end-to-end* y superando cuellos de botella algorítmicos.
# 
# ### 2.2. YOLO (You Only Look Once - One-Stage Detector)
# Introducido por Redmon et al. \cite{redmon2016you}, YOLO revolucionó el campo al plantear la detección como un único problema de regresión directa.
# 
# 1. La imagen de entrada se divide en una cuadrícula (*grid*) de $S \times S$ celdas.
# 2. Cada celda de la parrilla es responsable de predecir simultáneamente $B$ cuadros delimitadores, sus puntuaciones de confianza y las probabilidades condicionales de clase.
# 
# **Innovación clave:** Al procesar la imagen completa en una única pasada, alcanza velocidades en tiempo real e incorpora un contexto global, lo que reduce drásticamente los falsos positivos en el fondo.
# 
# ### 2.3. SSD (Single Shot MultiBox Detector - One-Stage Detector)
# Desarrollado por Liu et al. \cite{liu2016ssd}, SSD ofrece un compromiso estratégico entre la extrema velocidad de YOLO y la precisión analítica de Faster R-CNN.
# 
# * Aplica cabezales de detección sobre **múltiples mapas de características a diferentes escalas**.
# * Los mapas de capas más profundas (baja resolución) detectan objetos grandes, mientras que los mapas de capas más superficiales (alta resolución) detectan objetos pequeños.
# 
# #### Resumen Arquitectónico
# 
# | Arquitectura | Paradigma | Innovación Principal | Fortalezas / Debilidades |
# | :--- | :--- | :--- | :--- |
# | **Faster R-CNN** | Dos Etapas | Region Proposal Network (RPN) | **+** Muy alta precisión<br>**-** Lento computacionalmente |
# | **YOLO** | Una Etapa | Detección unificada basada en grid | **+** Extremadamente rápido<br>**-** Menos preciso en objetos pequeños |
# | **SSD** | Una Etapa | Detección multiescala con *anchor boxes* | **+** Buen balance velocidad-precisión<br>**-** Complejo de configurar |

# %% [markdown]
# ## 3. Anatomía General de un Detector Moderno
# 
# Independientemente del paradigma al que pertenezcan, las arquitecturas modernas han convergido en un diseño modular estandarizado que consta de tres componentes principales interconectados:
# 
# ### 3.1. Backbone (Extractor de Características)
# Es la red convolucional profunda (ej. ResNet, MobileNet, VGG) que actúa como un *encoder*. Toma la matriz de píxeles original y genera una jerarquía de mapas de características semánticamente ricos a diferentes escalas de resolución.
# 
# ### 3.2. Neck (Cuello - Fusión de Características)
# Un módulo estructural (siendo **FPN - Feature Pyramid Network** \cite{lin2017feature} el más destacado) que combina los mapas de características de diferentes niveles del *backbone*. Su objetivo es enriquecer las representaciones fusionando la **información semántica robusta** de las capas profundas con la **información espacial de alta resolución** de las capas superficiales, facilitando la detección multiescala.
# 
# ### 3.3. Head (Cabezal de Predicción)
# Toma los mapas de características optimizados por el *Neck* y ejecuta las predicciones finales. Estructuralmente se bifurca en dos ramas paralelas:
# * **Rama de Clasificación:** Predice la confianza ($s_i$) de que un cuadro contenga el objeto y a qué clase pertenece.
# * **Rama de Regresión:** Ajusta las coordenadas continuas del cuadro delimitador ($b_i$) para que se ciña con la mayor exactitud geométrica posible a la instancia real.

# %% [markdown]
# ## 4. Optimización: Funciones de Pérdida
# 
# Entrenar un detector de objetos es complejo porque la red debe aprender dos tareas de naturaleza distinta al mismo tiempo. Por ello, la pérdida total del modelo es siempre una suma ponderada:
# 
# $$ \mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda \cdot \mathcal{L}_{loc} $$
# 
# Donde $\mathcal{L}_{cls}$ penaliza los errores de clasificación, $\mathcal{L}_{loc}$ penaliza los errores geométricos de la caja delimitadora, y $\lambda$ es un hiperparámetro que equilibra ambas tareas.
# 
# ### 4.1. Pérdida de Clasificación ($\mathcal{L}_{cls}$)
# 
# El enfoque tradicional es usar la **Binary Cross-Entropy (BCE)**. Sin embargo, en detección enfrentamos un problema severo: en una imagen hay miles de cajas candidatas que son solo "fondo", y muy pocas que son objetos reales.
# 
# Para solucionar este desbalance, se utiliza la **Focal Loss** \cite{lin2017focal}. Esta función modifica la entropía cruzada añadiendo un factor de modulación que reduce el peso de los ejemplos "fáciles" (como el fondo claro) y obliga a la red a concentrarse en los objetos difíciles:
# 
# $$ \mathcal{L}_{FL}(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t) $$
# 
# Cuando el modelo está seguro de una predicción ($p_t \rightarrow 1$), el término $(1 - p_t)^\gamma$ se acerca a cero, anulando casi por completo la penalización de ese ejemplo.
# 
# ### 4.2. Pérdida de Localización ($\mathcal{L}_{loc}$)
# 
# Históricamente se utilizó la **Smooth L1 Loss**, ya que es menos sensible a valores atípicos (*outliers*) que una pérdida cuadrática estándar:
# 
# $$ \text{Smooth}_{L1}(x) = \begin{cases} 0.5 x^2 & \text{si } |x| < 1 \\ |x| - 0.5 & \text{en otro caso} \end{cases} $$
# 
# Sin embargo, la Smooth L1 optimiza cada coordenada (x, y, w, h) de forma independiente, lo cual no siempre se correlaciona con la métrica visual real de solapamiento.
# 
# Por ello, los modelos modernos utilizan **pérdidas basadas en IoU**. La más completa es la **CIoU Loss (Complete IoU)** \cite{zheng2020distance}, que penaliza tres aspectos geométricos simultáneamente:
# 1. La falta de solapamiento entre cajas.
# 2. La distancia normalizada entre los centros de las cajas ($\frac{\rho^2}{c^2}$).
# 3. La inconsistencia en la relación de aspecto ($\alpha v$).
# 
# $$ \mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b_p, b_{gt})}{c^2} + \alpha v $$

# %% [markdown]
# ## 5. Métricas de Evaluación en Detección
# 
# En clasificación, medir la "exactitud" (*accuracy*) es sencillo. En detección, necesitamos saber si el modelo predijo la clase correcta *y además* si la ubicó correctamente en el espacio.
# 
# ### 5.1. Intersection over Union (IoU)
# Es la métrica base. Mide geométricamente el solapamiento entre la caja predicha ($B_p$) y la caja real de *Ground Truth* ($B_{gt}$):
# 
# $$ \text{IoU} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})} $$
# 
# ### 5.2. Definiendo Aciertos y Errores
# Establecemos un **umbral de IoU** (generalmente 0.5) para decidir si una detección es válida:
# * **True Positive (TP):** La red predijo la clase correcta y el solapamiento con la caja real es mayor al umbral (ej. IoU > 0.5).
# * **False Positive (FP):** La red detectó un objeto donde no lo hay, o el solapamiento es menor al umbral.
# * **False Negative (FN):** Había un objeto real en la imagen, pero la red no lo detectó.
# 
# ### 5.3. Precision, Recall y mAP
# A partir de los TP, FP y FN, calculamos dos métricas fundamentales:
# * **Precision:** De todas las cajas que la red dibujó, ¿qué porcentaje era realmente un objeto? ($\frac{TP}{TP + FP}$).
# * **Recall:** De todos los objetos reales en la imagen, ¿qué porcentaje logró encontrar la red? ($\frac{TP}{TP + FN}$).
# 
# Como todo modelo arroja una "puntuación de confianza" por caja, podemos variar el umbral de aceptación para obtener múltiples pares de Precision y Recall. Esto forma una curva. El área bajo esta curva se conoce como **Average Precision (AP)**:
# 
# $$ \text{AP} = \int_0^1 p(r) \, dr $$
# 
# Finalmente, el **Mean Average Precision (mAP)** es simplemente el promedio del AP calculado para todas las clases del dataset. En competencias estrictas como MS COCO, se evalúa el **mAP@[0.5:0.95]**, que exige que el modelo sea preciso bajo múltiples niveles de exigencia de solapamiento.

# %% [markdown]
# # Parte II: Preparación del Entorno y Datos
# 
# En esta sección prepararemos toda la infraestructura de datos necesaria. Entrenar un detector requiere un manejo de tensores mucho más cuidadoso que un problema de clasificación, ya que cada imagen posee un número variable de objetos y, por ende, tensores de tamaño dinámico.

# %% [markdown]
# ## 6. Configuración del Entorno y Reproducibilidad
# 
# Comenzamos importando las librerías fundamentales y fijando las semillas de aleatoriedad. En investigación y docencia, garantizar que los experimentos sean reproducibles es innegociable.

# %%
import os
import random
import urllib.request
import zipfile
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDClassificationHead

# 1. Fijar semillas para reproducibilidad
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 2. Configurar dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de cómputo configurado: {device}")

# %% [markdown]
# ## 7. Adquisición y Preprocesamiento del Dataset (COCO128)
# 
# Para garantizar agilidad en la clase, utilizaremos **COCO128**, un subconjunto ligero de 128 imágenes del dataset original MS COCO. 
# 
# **Importante (Conversión Geométrica):** Las anotaciones en COCO/YOLO suelen venir en formato normalizado `[x_centro, y_centro, ancho, alto]`. Sin embargo, las arquitecturas de PyTorch (`torchvision`) exigen estrictamente coordenadas absolutas de las esquinas: `[x_min, y_min, x_max, y_max]`. Nuestro `Dataset` personalizado debe realizar esta transformación matemática al vuelo.

# %%
def download_coco128(root="./data"):
    """Descarga y extrae el dataset COCO128 automáticamente."""
    coco128_path = os.path.join(root, "coco128")
    if os.path.exists(coco128_path):
        print("Dataset COCO128 ya se encuentra localmente.")
        return coco128_path

    os.makedirs(root, exist_ok=True)
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = os.path.join(root, "coco128.zip")

    print("Descargando COCO128...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extrayendo archivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    os.remove(zip_path)
    return coco128_path

class COCO128Dataset(Dataset):
    """Dataset personalizado para procesar imágenes y bounding boxes."""
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs_path = os.path.join(root, "images", "train2017")
        self.labels_path = os.path.join(root, "labels", "train2017")
        self.imgs = sorted([f for f in os.listdir(self.imgs_path) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.imgs_path, img_name)
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.labels_path, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # Extraer formato YOLO: valores normalizados [0, 1]
                        x_center, y_center, width, height = map(float, parts[1:])

                        # Conversión matemática a formato Torchvision [x_min, y_min, x_max, y_max] absolutos
                        x_min = (x_center - width / 2) * img_width
                        y_min = (y_center - height / 2) * img_height
                        x_max = (x_center + width / 2) * img_width
                        y_max = (y_center + height / 2) * img_height

                        boxes.append([x_min, y_min, x_max, y_max])
                        # PyTorch reserva la clase 0 para el "background" (fondo)
                        labels.append(class_id + 1) 

        # Conversión a tensores de PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Manejo de casos límite: imágenes sin objetos de interés
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

# %% [markdown]
# ## 8. Construcción de los DataLoaders
# 
# Un aspecto crítico en detección es la función de colación (`collate_fn`). Por defecto, PyTorch intenta apilar los tensores en una matriz regular (ej. `[batch_size, num_objetos, 4]`). Dado que una imagen puede tener 2 objetos y otra 10, esto arrojaría un error. 
# 
# Para solucionarlo, reescribimos la función para que empaquete las muestras en tuplas, respetando la dimensionalidad variable de cada elemento del lote.

# %%
def collate_fn(batch):
    """Función personalizada para agrupar imágenes con distintos números de objetos."""
    return tuple(zip(*batch))

# Descargar y preparar dataset
dataset_path = download_coco128("./data")
transformaciones = transforms.Compose([transforms.ToTensor()])
dataset_completo = COCO128Dataset(dataset_path, transforms=transformaciones)

# Partición Estratégica (70% Train, 15% Val, 15% Test)
n_total = len(dataset_completo)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    dataset_completo, [n_train, n_val, n_test], 
    generator=torch.Generator().manual_seed(seed)
)

print(f"\nDistribución del Dataset: Train({len(train_ds)}) | Val({len(val_ds)}) | Test({len(test_ds)})")

batch_size = 4  # Lote pequeño debido a la carga de memoria que exige la detección
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# %% [markdown]
# ## 9. Definición del Modelo Base (SSDLite con MobileNetV3)
# 
# Para esta implementación, optamos por **SSDLite** respaldado por un *backbone* **MobileNetV3**.
# 
# **Justificación Pedagógica:** Entrenar un detector desde cero con pesos inicializados aleatoriamente requiere cientos de horas en clústers de GPUs para estabilizarse. Al utilizar un modelo preentrenado (*Transfer Learning*), aprovechamos la profunda jerarquía visual que el *backbone* ya extrajo de millones de imágenes. Esto nos permite enfocar nuestro esfuerzo computacional (y cognitivo) exclusivamente en el *Head* del modelo, logrando convergencia rápida y validando de manera práctica los conceptos aprendidos.

# %%
def get_object_detection_model(num_classes):
    """
    Carga un modelo SSDLite preentrenado y ajusta el cabezal de clasificación.
    num_classes debe incluir el fondo (background) como la clase 0.
    """
    # 1. Cargar el modelo preentrenado en COCO
    # Usamos weights="DEFAULT" para obtener los mejores pesos disponibles en torchvision
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    
    # 2. Modificación del Cabezal (Head)
    # Por defecto, el modelo preentrenado tiene 91 clases (90 COCO + 1 fondo).
    # Aunque COCO128 usa clases similares, esta es la forma estándar de adaptar 
    # la arquitectura si trajéramos nuestro propio dataset personalizado (ej. detectar solo melanomas).
    
    # Extraemos el número de anclas (anchor boxes) por ubicación espacial
    in_channels = torchvision.models.detection._utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # Reemplazamos la capa de clasificación para que coincida con nuestras clases
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model

# COCO define 80 clases de objetos. PyTorch requiere sumar 1 clase adicional para el 'background'.
NUM_CLASSES = 81 
model = get_object_detection_model(NUM_CLASSES)
model = model.to(device)

print("\nArquitectura SSDLite instanciada y enviada al dispositivo correctamente.")

# %% [markdown]
# # Parte III: Entrenamiento, Evaluación y Análisis Geométrico
# 
# Si bien estamos utilizando una arquitectura SSDLite preentrenada para garantizar la viabilidad computacional del entrenamiento, la evaluación de un detector no debe ser una "caja negra". 
# 
# En esta sección, programaremos desde cero las matemáticas del *Intersection over Union (IoU)* y el cálculo de la precisión media para entender exactamente cómo se castiga o premia al modelo por sus predicciones espaciales.

# %% [markdown]
# ## 10. Implementación Manual de Métricas de Evaluación
# 
# La base de cualquier evaluación en detección es calcular el solapamiento entre la predicción y el *Ground Truth*.

# %%
def compute_iou_boxes(box1, box2):
    """
    Calcula geométricamente el Intersection over Union (IoU) entre dos cajas.
    Formato esperado: [x_min, y_min, x_max, y_max]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 1. Coordenadas de la Intersección
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # El cálculo de max(0, ...) evita áreas negativas si las cajas no se tocan
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 2. Área de la Unión
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return float(inter_area / union_area)

def compute_map_simplified(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Calcula una versión simplificada de mAP, Precision y Recall iterando
    sobre las predicciones y buscando emparejamientos válidos.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().detach()
        pred_labels = pred['labels'].cpu().detach()
        pred_scores = pred['scores'].cpu().detach()

        target_boxes = target['boxes'].cpu().detach()
        target_labels = target['labels'].cpu().detach()

        # Filtrar el ruido: predicciones con baja confianza
        mask = pred_scores > score_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        matched_targets = set()

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_idx = -1

            # Buscar el mejor emparejamiento en el Ground Truth
            for idx, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
                if idx in matched_targets:
                    continue # Este objeto real ya fue detectado por otra caja

                if pred_label == target_label:
                    iou = compute_iou_boxes(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

            # Clasificación topológica del resultado
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_targets.add(best_idx)
            else:
                total_fp += 1

        # Los objetos reales que nunca fueron emparejados son Falsos Negativos
        total_fn += len(target_boxes) - len(matched_targets)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

# %% [markdown]
# ## 11. Pipeline de Entrenamiento (Training Loop)
# 
# Entrenar un detector requiere manejar un diccionario de pérdidas. A diferencia de la clasificación simple, PyTorch nos devolverá un sumatorio de las pérdidas de regresión y clasificación de la arquitectura.

# %%
import torch.optim as optim

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc="Entrenando")
    for images, targets in progress_bar:
        # Trasladar datos a la GPU/CPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward Pass: En modo train, los modelos de detección en torchvision 
        # devuelven directamente el diccionario de pérdidas, no las predicciones.
        loss_dict = model(images, targets)
        
        # La pérdida total es la suma ponderada de todas las sub-pérdidas
        losses = sum(loss for loss in loss_dict.values())

        # Backward Pass y Optimización
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        progress_bar.set_postfix({'loss': f"{losses.item():.4f}"})
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval() # Fundamental: Cambia el comportamiento de Dropout y BatchNorm, y hace que el modelo devuelva predicciones
    all_predictions = []
    all_targets = []

    for images, targets in loader:
        images = list(image.to(device) for image in images)
        predictions = model(images)
        all_predictions.extend(predictions)
        all_targets.extend(targets)

    return compute_map_simplified(all_predictions, all_targets)

# Configuración de Hiperparámetros
n_epochs = 5
params = [p for p in model.parameters() if p.requires_grad]
# Usamos SGD con Momentum, que históricamente estabiliza los gradientes complejos en detección
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Bucle principal
history = {'train_loss': [], 'val_f1': []}
best_f1 = 0.0

print("\n" + "="*50)
print("INICIANDO ENTRENAMIENTO SSDLite")
print("="*50)

for epoch in range(n_epochs):
    print(f"\nÉpoca {epoch+1}/{n_epochs}")
    
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_metrics = evaluate_model(model, val_loader, device)
    
    lr_scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['val_f1'].append(val_metrics['f1_score'])
    
    print(f"Pérdida Entrenamiento: {train_loss:.4f}")
    print(f"Métricas Val -> Precision: {val_metrics['precision']:.3f} | Recall: {val_metrics['recall']:.3f} | F1: {val_metrics['f1_score']:.3f}")
    
    if val_metrics['f1_score'] > best_f1:
        best_f1 = val_metrics['f1_score']
        torch.save(model.state_dict(), "best_ssdlite.pth")
        print(" -> Mejor modelo guardado en disco.")

# %% [markdown]
# ## 12. Análisis y Visualización de Resultados
# 
# Finalmente, cargamos los pesos del mejor modelo y evaluamos su capacidad de generalización sobre el conjunto de pruebas que nunca ha visto, visualizando gráficamente dónde acierta y dónde falla.

# %%
def visualize_predictions(model, dataset, device, num_samples=4):
    model.load_state_dict(torch.load("best_ssdlite.pth", map_location=device))
    model.eval()
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, target = dataset[idx]
            prediction = model([img.to(device)])[0]
            
            ax = axes[i]
            # Des-normalizar la imagen para Matplotlib (C, H, W) -> (H, W, C)
            ax.imshow(img.permute(1, 2, 0).cpu().numpy())
            
            # Dibujar Ground Truth (Verde)
            for box in target['boxes']:
                xmin, ymin, xmax, ymax = box.numpy()
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                         linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                
            # Dibujar Predicciones (Rojo)
            scores = prediction['scores'].cpu().numpy()
            boxes = prediction['boxes'].cpu().numpy()
            
            for box, score in zip(boxes, scores):
                if score > 0.5: # Umbral de confianza
                    xmin, ymin, xmax, ymax = box
                    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                             linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(xmin, ymin-5, f"{score:.2f}", color='white', 
                            bbox=dict(facecolor='red', alpha=0.5), fontsize=8)
            
            ax.axis('off')
            ax.set_title(f"Muestra de Prueba {i+1}")
            
    plt.suptitle("Verde (--): Ground Truth | Rojo (—): Predicciones SSDLite", fontsize=14)
    plt.tight_layout()
    plt.show()

# Ejecutar visualización
visualize_predictions(test_ds.dataset, device)