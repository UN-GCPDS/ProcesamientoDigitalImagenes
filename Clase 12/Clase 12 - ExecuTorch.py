# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# <img src="https://www.funcionpublica.gov.co/documents/d/guest/logo-universidad-nacional" alt="Logo UNAL" width="600"/>
#
# ### **Universidad Nacional de Colombia sede Manizales**
# #### Facultad de ingeniería y arquitectura
# #### Departamento de ingeniería eléctrica, electrónica y computación
# #### *Procesamiento Digital de Imágenes*
#
# #### Profesor: Lucas Iturriago
# #### Monitora: Isabella Valero Mora - lvalerom@unal.edu.co

# %% [markdown]
# # 1. Exportación a Float32 con el Ecosistema de ExecuTorch
#
# En el procesamiento digital de imágenes y visión por computador en dispositivos de borde (edge), la optimización del tamaño del modelo y la velocidad de inferencia son de vital importancia. ExecuTorch es la plataforma insignia de PyTorch para ejecutar modelos directamente en teléfonos móviles, dispositivos IoT y hardware de propósito general de forma altamente eficiente.
#
#
# En esta clase exportaremos los modelos en su precisión estándar de **Float32 (precisión simple)**, la cual está altamente optimizada para CPUs modernas gracias a conjuntos de instrucciones vectoriales (como AVX o NEON) y garantiza una fidelidad del 100% en las predicciones.
#
# ## 1.1. Ciclo de Vida de Exportación de ExecuTorch
#
# El pipeline completo de exportación consta de los siguientes pasos:
# 1.  **Captura del Grafo (torch.export):** Se rastrea el flujo de ejecución de PyTorch y se genera un grafo en el dialecto de operadores de ATen (`ExportedProgram`).
# 2.  **Conversión a Edge (EXIR Edge Dialect):** Se transforma el grafo a un conjunto simplificado de operadores optimizados para dispositivos de borde.
# 3.  **Particionado y Delegación (XNNPACK):** Se identifican subgrafos que pueden ser acelerados por la biblioteca de alto rendimiento XNNPACK para CPU.
# 4.  **Serialización (.pte):** Se guarda el grafo resultante como un archivo Flatbuffer con extensión `.pte` listo para su ejecución.

# %%
# !pip install executorch -q
# !pip install torch==2.11.0 -q
# !pip install ultralytics -q
import torch
import torchvision.models as models
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# %% [markdown]
# ## 1.2. Configuración de Parámetros Globales de Entrada
#
# Definimos las dimensiones de entrada requeridas para compilar cada uno de los modelos de nuestra sesión en formato `torch.float32`.

# %%
# Dimensiones de entrada en formato (Batch, Canales, Alto, Ancho)
CLASSIFICATION_INPUT_SIZE = (1, 3, 224, 224)
SEGMENTATION_INPUT_SIZE = (1, 3, 256, 256)
DETECTION_INPUT_SIZE = (1, 3, 640, 640)

# %% [markdown]
# # 2. Exportación del Modelo de Clasificación (MobileNetV2)
#
# MobileNetV2 es una arquitectura liviana y eficiente que utiliza convoluciones separables en profundidad.
#
# $$y = \text{MobileNetV2}(x), \quad x \in \mathbb{R}^{1 \times 3 \times 224 \times 224}$$
#
# Cargamos el modelo preentrenado, lo pasamos a modo evaluación en precisión `float32`, y creamos los tensores de entrada de prueba correspondientes.

# %%
print("--- [1] EXPORTANDO CLASIFICACIÓN (MobileNetV2 FP32) ---")

# 1. Cargar el modelo preentrenado en modo evaluación
model_cls = models.mobilenet_v2(pretrained=True).eval()

# 2. Crear una entrada de prueba en Float32
sample_input_cls = (torch.randn(CLASSIFICATION_INPUT_SIZE),)

# 3. Capturar el grafo ATen usando torch.export
exported_cls = export(model_cls, sample_input_cls)

# 4. Transformar a Edge Dialect y bajar a XNNPACK con soporte FP32
edge_program_cls = to_edge_transform_and_lower(
    exported_cls,
    partitioner=[XnnpackPartitioner()]
)

# 5. Serializar el modelo a formato .pte
executorch_program_cls = edge_program_cls.to_executorch()
with open("mobilenet_v2.pte", "wb") as f:
    f.write(executorch_program_cls.buffer)

print("Clasificación: Modelo guardado con éxito como 'mobilenet_v2.pte'")

# %% [markdown]
# # 3. Exportación del Modelo de Segmentación Semántica (DeepLabV3)
#
# DeepLabV3 es un modelo avanzado de segmentación de imágenes que utiliza convoluciones atrosas para extraer características multiescala.
#
# Creamos un módulo contenedor `SegmentationWrapper` para retornar únicamente el tensor principal de predicciones de segmentación, resolviendo incompatibilidades con diccionarios de salida:
#
# $$\hat{y} = \text{DeepLabV3}(x)[\text{"out"}] \in \mathbb{R}^{1 \times C_{classes} \times H \times W}$$

# %%
print("\n--- [2] EXPORTANDO SEGMENTACIÓN (DeepLabV3 FP32) ---")

# 1. Definir el wrapper para resolver incompatibilidades de diccionario en la exportación
class SegmentationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs["out"]

# 2. Cargar el modelo de segmentación preentrenado (DeepLabV3 con backbone MobileNetV3)
raw_segmentation_model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()

# 3. Instanciar wrapper en Float32
model_seg = SegmentationWrapper(raw_segmentation_model)

# 4. Crear entrada de prueba en Float32
sample_input_seg = (torch.randn(SEGMENTATION_INPUT_SIZE),)

# 5. Exportar el modelo
exported_seg = export(model_seg, sample_input_seg)

# 6. Transformar y bajar a XNNPACK
edge_program_seg = to_edge_transform_and_lower(
    exported_seg,
    partitioner=[XnnpackPartitioner()]
)

# 7. Serializar a formato .pte
executorch_program_seg = edge_program_seg.to_executorch()
with open("deeplabv3.pte", "wb") as f:
    f.write(executorch_program_seg.buffer)

print("Segmentación: Modelo guardado con éxito como 'deeplabv3.pte'")

# %% [markdown]
# # 4. Exportación del Modelo de Detección de Objetos (YOLO26)
#
# YOLO26 es una arquitectura avanzada y eficiente de Ultralytics lanzada en enero de 2026.
# Cuenta con un diseño nativamente NMS-free que elimina la necesidad de postprocesamiento heurístico en entrenamiento.
#
# Para exportarlo de forma segura a ExecuTorch, usaremos el método `.export` provisto directamente por la librería de Ultralytics.

# %%
print("\n--- [3] EXPORTANDO DETECCIÓN DE OBJETOS (YOLO26 FP32) ---")

from ultralytics import YOLO
import shutil
import os

# 1. Cargar el detector de objetos YOLO26 Nano preentrenado
model_det = YOLO("yolo26n.pt")

# 2. Exportar a formato ExecuTorch
# El método .export() devuelve la ruta de la carpeta con el modelo exportado
export_dir = model_det.export(format="executorch")

# 3. Copiar el modelo exportado a la ubicación raíz esperada
shutil.copy(os.path.join(export_dir, "model.pte"), "yolo26.pte")

print("Detección de Objetos: Modelo guardado con éxito como 'yolo26.pte'")
