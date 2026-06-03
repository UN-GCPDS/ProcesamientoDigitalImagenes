# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # 1. Cuantización a Float16 y el Ecosistema de ExecuTorch
# 
# En el procesamiento digital de imágenes y visión por computador en dispositivos de borde (edge), la optimización del tamaño del modelo y la velocidad de inferencia son de vital importancia. ExecuTorch es la plataforma insignia de PyTorch para ejecutar modelos directamente en teléfonos móviles, dispositivos IoT y hardware especializado de forma altamente eficiente.
# 
# Uno de los métodos fundamentales de compresión es la conversión o **cuantización a precisión float16 (media precisión)**. Mientras que la precisión estándar de entrenamiento es float32 (precisión simple), reducirla a float16 reduce el tamaño de almacenamiento del modelo a la mitad y duplica el rendimiento del ancho de banda de memoria.
# 
# ## 1.1. Representación Matemática de Float16 (IEEE 754)
# 
# Un número flotante de media precisión (float16) consta de 16 bits en total, distribuidos de la siguiente manera:
# *   **1 bit de signo ($S$)**
# *   **5 bits de exponente ($E$)** con un sesgo (bias) de $15$
# *   **10 bits de mantisa o fracción ($M$)**
# 
# La ecuación matemática que define su valor real está dada por:
# 
# $$V = (-1)^S \times 2^{E - 15} \times \left(1 + \sum_{i=1}^{10} b_i 2^{-i}\right)$$
# 
# Donde $b_i \in \{0, 1\}$ son los coeficientes binarios de la mantisa. En comparación, un número de precisión simple (float32) utiliza 32 bits (1 de signo, 8 de exponente con sesgo de 127, y 23 de mantisa):
# 
# $$V = (-1)^S \times 2^{E - 127} \times \left(1 + \sum_{i=1}^{23} b_i 2^{-i}\right)$$
# 
# Al limitar la mantisa y el rango del exponente, podemos representar valores en un intervalo de aproximadamente $\pm 65504$ con una precisión de hasta $0.000977$. Esto es más que suficiente para la mayoría de arquitecturas de visión profunda en fase de inferencia.
# 
# ## 1.2. Ciclo de Vida de Exportación de ExecuTorch
# 
# El pipeline completo de exportación consta de los siguientes pasos:
# 1.  **Captura del Grafo (torch.export):** Se rastrea el flujo de ejecución de PyTorch y se genera un grafo en el dialecto de operadores de ATen (`ExportedProgram`).
# 2.  **Conversión a Edge (EXIR Edge Dialect):** Se transforma el grafo a un conjunto simplificado de operadores optimizados para dispositivos móviles.
# 3.  **Particionado y Delegación (XNNPACK/CoreML):** Se identifican subgrafos que pueden ser acelerados por aceleradores de hardware como XNNPACK (para CPU ARM/x86) o CoreML (Apple Silicon) aplicando precisiones reducidas.
# 4.  **Serialización (.pte):** Se guarda el grafo resultante como un archivo Flatbuffer con extensión `.pte` listo para su ejecución.

# %%
import torch
import torchvision.models as models
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# %% [markdown]
# ## 1.3. Configuración de Parámetros Globales de Entrada
# 
# Definimos las dimensiones y tipos de datos requeridos para compilar cada uno de los modelos de nuestra sesión.
# En este caso, todas nuestras entradas y pesos serán forzados a precisión `torch.float16`.

# %%
# Dimensiones de entrada en formato (Batch, Canales, Alto, Ancho)
CLASSIFICATION_INPUT_SIZE = (1, 3, 224, 224)
SEGMENTATION_INPUT_SIZE = (1, 3, 256, 256)
DETECTION_INPUT_SIZE = (1, 3, 320, 320)

# %% [markdown]
# # 2. Exportación del Modelo de Clasificación (MobileNetV2)
# 
# MobileNetV2 es una arquitectura liviana y eficiente que utiliza convoluciones separables en profundidad.
# 
# $$y = \text{MobileNetV2}(x), \quad x \in \mathbb{R}^{1 \times 3 \times 224 \times 224}$$
# 
# Para exportar en float16, primero cargamos el modelo preentrenado, lo pasamos a modo evaluación, lo casteamos a float16 usando `.half()`, y creamos tensores de entrada de prueba con el mismo tipo de precisión.

# %%
print("--- [1] EXPORTANDO CLASIFICACIÓN (MobileNetV2 FP16) ---")

# 1. Cargar el modelo preentrenado en modo evaluación
model_cls = models.mobilenet_v2(pretrained=True).eval()

# 2. Convertir el modelo completo a Float16
model_cls_fp16 = model_cls.half()

# 3. Crear una entrada de prueba del tipo de datos float16
sample_input_cls = (torch.randn(CLASSIFICATION_INPUT_SIZE).half(),)

# 4. Capturar el grafo ATen usando torch.export
exported_cls = export(model_cls_fp16, sample_input_cls)

# 5. Transformar a Edge Dialect y bajar a XNNPACK con soporte FP16
edge_program_cls = to_edge_transform_and_lower(
    exported_cls,
    partitioner=[XnnpackPartitioner()]
)

# 6. Serializar el modelo a formato .pte
executorch_program_cls = edge_program_cls.to_executorch()
with open("mobilenet_v2_fp16.pte", "wb") as f:
    f.write(executorch_program_cls.buffer)

print("Clasificación: Modelo guardado con éxito como 'mobilenet_v2_fp16.pte'")

# %% [markdown]
# # 3. Exportación del Modelo de Segmentación Semántica (DeepLabV3)
# 
# DeepLabV3 es un modelo avanzado de segmentación de imágenes que utiliza convoluciones atrosas (con dilatación espacial) para extraer características multiescala.
# 
# Los modelos de segmentación en TorchVision devuelven diccionarios que contienen múltiples salidas intermedias (`"out"` y `"aux"`). Sin embargo, `torch.export` requiere salidas simplificadas en forma de tensores o tuplas de tensores.
# 
# Creamos un módulo contenedor `SegmentationWrapper` para retornar únicamente el tensor principal de predicciones de segmentación:
# 
# $$\hat{y} = \text{DeepLabV3}(x)[\text{"out"}] \in \mathbb{R}^{1 \times C_{classes} \times H \times W}$$

# %%
print("\n--- [2] EXPORTANDO SEGMENTACIÓN (DeepLabV3 FP16) ---")

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

# 3. Instanciar wrapper y convertir pesos a Float16
model_seg_fp16 = SegmentationWrapper(raw_segmentation_model).half()

# 4. Crear entrada de prueba en float16
sample_input_seg = (torch.randn(SEGMENTATION_INPUT_SIZE).half(),)

# 5. Exportar el modelo
exported_seg = export(model_seg_fp16, sample_input_seg)

# 6. Transformar y bajar a XNNPACK
edge_program_seg = to_edge_transform_and_lower(
    exported_seg,
    partitioner=[XnnpackPartitioner()]
)

# 7. Serializar a formato .pte
executorch_program_seg = edge_program_seg.to_executorch()
with open("deeplabv3_fp16.pte", "wb") as f:
    f.write(executorch_program_seg.buffer)

print("Segmentación: Modelo guardado con éxito como 'deeplabv3_fp16.pte'")

# %% [markdown]
# # 4. Exportación del Modelo de Detección de Objetos (SSDLite)
# 
# El modelo SSDLite es una variante compacta de SSD (Single Shot MultiBox Detector) enfocada en dispositivos embebidos.
# 
# Los modelos de detección de TorchVision son dinámicos: realizan internamente el escalamiento de imágenes y ejecutan operaciones complejas de filtrado no máximo (Non-Maximum Suppression - NMS) que varían el tamaño de salida según el número de objetos detectados.
# 
# Para exportarlo de forma segura con `torch.export` a ExecuTorch, crearemos un Wrapper llamado `SSDLiteWrapper`. Este wrapper procesará las características y las pasará por la cabeza de predicción (`head`) de forma directa, eludiendo la lógica interna y retornando los tensores planos y crudos (raw outputs) de las cajas candidatas de regresión y las puntuaciones de las clases correspondientes.
# 
# Las salidas del wrapper corresponden a:
# 1.  `bbox_regression`: Delimitación espacial tentativa.
# 2.  `cls_logits`: Distribución de probabilidad sobre las clases de los objetos.
# 
# El post-procesamiento (NMS y decodificación) se trasladará al cliente para optimizar la eficiencia y portabilidad en el backend de inferencia.

# %%
print("\n--- [3] EXPORTANDO DETECCIÓN DE OBJETOS (SSDLite FP16) ---")

# 1. Definir el wrapper para extraer las predicciones brutas de la red neuronal
class SSDLiteWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Extraer características usando la red troncal (backbone)
        features = self.model.backbone(x)
        features = list(features.values())
        
        # Calcular los scores y offsets de regresión de cajas de anclas a través de la cabeza (head)
        head_outputs = self.model.head(features)
        
        # Retornamos los tensores crudos para que no haya control de flujo dinámico durante la exportación
        return head_outputs["bbox_regression"], head_outputs["cls_logits"]

# 2. Cargar el detector de objetos SSDLite con MobileNetV3 Large preentrenado
raw_detection_model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True).eval()

# 3. Instanciar wrapper y convertir pesos a Float16
model_det_fp16 = SSDLiteWrapper(raw_detection_model).half()

# 4. Crear entrada de prueba en float16 (320x320 píxeles)
sample_input_det = (torch.randn(DETECTION_INPUT_SIZE).half(),)

# 5. Exportar el modelo
exported_det = export(model_det_fp16, sample_input_det)

# 6. Transformar y bajar a XNNPACK
edge_program_det = to_edge_transform_and_lower(
    exported_det,
    partitioner=[XnnpackPartitioner()]
)

# 7. Serializar a formato .pte
executorch_program_det = edge_program_det.to_executorch()
with open("ssdlite_fp16.pte", "wb") as f:
    f.write(executorch_program_det.buffer)

print("Detección de Objetos: Modelo guardado con éxito como 'ssdlite_fp16.pte'")
