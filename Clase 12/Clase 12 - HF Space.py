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

# %% [markdown]
# # 1. Despliegue de Modelos ExecuTorch en Hugging Face Spaces (Gradio SDK)
#
# Una vez exportados los modelos al formato optimizado de ExecuTorch (`.pte`), el siguiente paso es desplegarlos en producción para que puedan ser consumidos mediante una interfaz web o como API externa. **Hugging Face Spaces** permite alojar aplicaciones interactivas usando de forma nativa el **SDK de Gradio** (`sdk: gradio`).
#
# ## 1.1. Arquitectura del Despliegue
#
# En esta sesión construiremos un servidor Gradio multimodelo capaz de ejecutar inferencia con ExecuTorch en precisión Float32 para tres tareas de visión por computador:
# 1. **Clasificación de imágenes** (`mobilenet_v2.pte`)
# 2. **Segmentación semántica** (`deeplabv3.pte`)
# 3. **Detección de objetos** (`yolo26.pte`)
#
# ### Componentes del Space:
# * **Frontend (`app.py`):** Aplicación en `gr.Blocks` organizada por pestañas, con decorador `@spaces.GPU` (compatible tanto con CPU estándar como con ZeroGPU) y rutas `api_name` explícitas para consumo remoto.
# * **Dependencias (`requirements.txt`):** Gestión declarativa directa sobre el SDK nativo de Gradio (`sdk: gradio`), sin necesidad de construir manualmente un `Dockerfile`.
# * **Publicación (`HfApi`):** Creación del Space y carga directa de los archivos y modelos `.pte` mediante `HfApi.create_repo` y `HfApi.upload_folder`, evitando configuraciones manuales de `git clone`, `git lfs` o `git push`.

# %% [markdown]
# ## 1.2. Instalación de Dependencias del Entorno
#
# Instalamos `gradio`, `gradio_client` y `huggingface_hub` para empaquetar, publicar y probar el Space directamente desde el cuaderno.

# %%
!pip install gradio gradio_client huggingface_hub -q

# %% [markdown]
# # 2. Configuración y Autenticación en Hugging Face
#
# ## 2.1. Autenticación en Hugging Face
#
# Ejecuta la siguiente celda e ingresa tu **User Access Token** (con permisos de *write*) obtenido en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

# %%
from huggingface_hub import notebook_login

# Inicia sesión en tu cuenta de Hugging Face
try:
    notebook_login()
except Exception as e:
    print(f"[INFO] notebook_login no está disponible en este entorno: {e}")

# %% [markdown]
# ## 2.2. Parámetros del Space
#
# Configura tu nombre de usuario de Hugging Face, el nombre del Space y el directorio local donde empaquetaremos los archivos antes de subirlos.

# %%
from pathlib import Path

HF_USER = "tu_usuario_hf"  # <-- REEMPLAZA CON TU USUARIO DE HUGGING FACE
SPACE_NAME = "executorch-pdi-unal"
SPACE_DIR = Path(SPACE_NAME)
SPACE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Space configurado: https://huggingface.co/spaces/{HF_USER}/{SPACE_NAME}")
print(f"Directorio local de empaquetado: {SPACE_DIR.resolve()}")

# %% [markdown]
# # 3. Generación de los Archivos de Despliegue (Gradio SDK)
#
# Generaremos localmente los archivos requeridos por Hugging Face Spaces dentro de `SPACE_DIR`:
# 1. `app.py`: Servidor Gradio conectado al runtime nativo de ExecuTorch.
# 2. `requirements.txt`: Dependencias instaladas automáticamente por el SDK de Gradio.
# 3. `README.md`: Metadatos YAML con `sdk: gradio` y la versión exacta de Gradio instalada.

# %% [markdown]
# ## 3.1. Generación de `app.py` (Gradio con Runtime de ExecuTorch)
#
# Escribimos el código de la aplicación web que cargará los archivos `.pte` mediante `executorch.runtime.Runtime` y expondrá cada tarea como una pestaña y endpoint de API.

# %%
app_code = r"""import os
import urllib.request
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

# -------------------------------------------------------------
# 0. Soporte para Entornos con ZeroGPU (Hugging Face Spaces)
# -------------------------------------------------------------
try:
    import spaces
    print("[INFO] Módulo 'spaces' (ZeroGPU) detectado y cargado.")
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func

            def decorator(f):
                return f

            return decorator

    print("[INFO] Ejecutando en entorno estándar CPU / No-ZeroGPU.")

# -------------------------------------------------------------
# 1. Carga de Librerías y Runtime ExecuTorch
# -------------------------------------------------------------
try:
    from executorch.runtime import Runtime
    print("[INFO] Runtime nativo de ExecuTorch cargado exitosamente.")
except ImportError as exc:
    raise ImportError(
        "[ERROR] No se encontró el runtime de ExecuTorch. Verifica que 'executorch' esté en requirements.txt"
    ) from exc

# -------------------------------------------------------------
# 2. Constantes, Clases y Funciones de Pre/Postprocesamiento
# -------------------------------------------------------------
PATH_MODEL_CLS = "mobilenet_v2.pte"
PATH_MODEL_SEG = "deeplabv3.pte"
PATH_MODEL_DET = "yolo26.pte"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

rng = np.random.default_rng(42)
COLOR_PALETTE = rng.integers(0, 255, size=(100, 3), dtype=np.uint8)
COLOR_PALETTE[0] = np.array([0, 0, 0], dtype=np.uint8)

try:
    IMAGENET_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(IMAGENET_URL, timeout=5) as response:
        imagenet_classes = response.read().decode("utf-8").splitlines()
except Exception:
    imagenet_classes = [f"Clase {i}" for i in range(1000)]


def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _error_image(image: Image.Image, message: str) -> Image.Image:
    img_np = np.array(_to_rgb(image))
    cv2.putText(img_np, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return Image.fromarray(img_np)


def preprocesar_imagen(img_pil: Image.Image, size: tuple, normalizar: bool = True) -> torch.Tensor:
    img_rgb = _to_rgb(img_pil)
    img_resized = img_rgb.resize(size)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    if normalizar:
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_transposed = np.transpose(img_np, (2, 0, 1))
    return torch.from_numpy(img_transposed).unsqueeze(0).contiguous()


def postprocesar_segmentacion(output_tensor: torch.Tensor, original_img: Image.Image) -> Image.Image:
    img_rgb = _to_rgb(original_img)
    mask = torch.argmax(output_tensor[0], dim=0).detach().cpu().numpy().astype(np.uint8)
    w, h = img_rgb.size
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_color = COLOR_PALETTE[mask_resized % len(COLOR_PALETTE)]
    img_orig_np = np.array(img_rgb)
    blended = cv2.addWeighted(img_orig_np, 0.6, mask_color, 0.4, 0)
    return Image.fromarray(blended)


def postprocesar_deteccion(boxes, scores, labels, original_img: Image.Image, threshold: float = 0.25) -> Image.Image:
    img_np = np.array(_to_rgb(original_img))
    h, w, _ = img_np.shape

    boxes_np = _to_numpy(boxes)
    scores_np = _to_numpy(scores)
    labels_np = _to_numpy(labels)

    for box, score, label_idx in zip(boxes_np, scores_np, labels_np):
        if score >= threshold:
            xmin = max(0, min(int(box[0]), w - 1))
            ymin = max(0, min(int(box[1]), h - 1))
            xmax = max(0, min(int(box[2]), w - 1))
            ymax = max(0, min(int(box[3]), h - 1))

            cls_idx = int(label_idx) % len(COCO_CLASSES)
            label_text = f"{COCO_CLASSES[cls_idx]}: {float(score):.2f}"
            color = [int(c) for c in COLOR_PALETTE[int(label_idx) % len(COLOR_PALETTE)]]

            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), color, 3)
            cv2.putText(img_np, label_text, (xmin, max(ymin - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return Image.fromarray(img_np)


# -------------------------------------------------------------
# 3. Gestor de Inferencia (ExecuTorch FP32)
# -------------------------------------------------------------
class ExecuTorchRunner:
    def __init__(self, pte_path: str):
        self.pte_path = pte_path
        self.method = None
        self.error = None

        if not os.path.exists(pte_path):
            self.error = f"Modelo no encontrado: {pte_path}"
            print(f"[WARNING] {self.error}")
            return

        try:
            print(f"[*] Cargando modelo ExecuTorch: {pte_path}")
            self.runtime = Runtime.get()
            self.program = self.runtime.load_program(pte_path)
            self.method = self.program.load_method("forward")
            print(f"[+] Modelo ExecuTorch inicializado correctamente: {pte_path}")
        except Exception as e:
            self.error = f"Error al cargar '{pte_path}': {e}"
            print(f"[ERROR] {self.error}")

    @property
    def is_ready(self) -> bool:
        return self.method is not None

    def predict(self, input_tensor: torch.Tensor):
        if not self.is_ready:
            raise RuntimeError(self.error or f"Modelo '{self.pte_path}' no disponible.")
        outputs = self.method.execute([input_tensor])
        if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            return outputs[0]
        return outputs


runner_cls = ExecuTorchRunner(PATH_MODEL_CLS)
runner_seg = ExecuTorchRunner(PATH_MODEL_SEG)
runner_det = ExecuTorchRunner(PATH_MODEL_DET)


# -------------------------------------------------------------
# 4. Funciones de Predicción para la Interfaz (@spaces.GPU)
# -------------------------------------------------------------
@spaces.GPU
def predict_classification(image: Image.Image) -> dict:
    if image is None:
        return {}
    if not runner_cls.is_ready:
        return {"Error": 1.0, runner_cls.error: 0.0}

    try:
        tensor = preprocesar_imagen(image, (224, 224))
        output = runner_cls.predict(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 5)
        return {imagenet_classes[int(idx)]: float(prob) for prob, idx in zip(top_prob, top_catid)}
    except Exception as e:
        return {"Error": 1.0, str(e): 0.0}


@spaces.GPU
def predict_segmentation(image: Image.Image) -> Image.Image:
    if image is None:
        return None
    if not runner_seg.is_ready:
        return _error_image(image, runner_seg.error)

    try:
        tensor = preprocesar_imagen(image, (256, 256))
        output = runner_seg.predict(tensor)
        if isinstance(output, dict):
            output_tensor = output["out"]
        elif isinstance(output, (list, tuple)):
            output_tensor = output[0]
        else:
            output_tensor = output

        if isinstance(output_tensor, np.ndarray):
            output_tensor = torch.from_numpy(output_tensor)
        return postprocesar_segmentacion(output_tensor, image)
    except Exception as e:
        return _error_image(image, f"Error: {e}")


@spaces.GPU
def predict_detection(image: Image.Image) -> Image.Image:
    if image is None:
        return None
    if not runner_det.is_ready:
        return _error_image(image, runner_det.error)

    try:
        tensor = preprocesar_imagen(image, (640, 640), normalizar=False)
        output = runner_det.predict(tensor)
        raw = output[0] if isinstance(output, (list, tuple)) else output
        pred = _to_numpy(raw)
        if pred.ndim == 3:
            pred = pred[0]

        orig_w, orig_h = image.size
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0
        conf_threshold = 0.25

        # Formato NMS-free end-to-end (YOLO26): (N, 6) -> [x1, y1, x2, y2, conf, class_id]
        if pred.ndim == 2 and pred.shape[1] == 6:
            dets = pred[pred[:, 4] >= conf_threshold]
            if len(dets) > 0:
                boxes_xyxy = dets[:, :4].copy()
                boxes_xyxy[:, [0, 2]] *= scale_x
                boxes_xyxy[:, [1, 3]] *= scale_y
                return postprocesar_deteccion(boxes_xyxy, dets[:, 4], dets[:, 5], image, threshold=conf_threshold)
            return _to_rgb(image)

        # Formato estándar sin NMS: (4 + C, N) -> transponer a (N, 4 + C)
        predictions = pred.T
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        keep = max_scores >= conf_threshold
        filtered_boxes = boxes[keep]
        filtered_scores = max_scores[keep]
        filtered_class_ids = class_ids[keep]

        if len(filtered_boxes) > 0:
            cx, cy, w_box, h_box = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
            x1 = (cx - w_box / 2) * scale_x
            y1 = (cy - h_box / 2) * scale_y
            x2 = (cx + w_box / 2) * scale_x
            y2 = (cy + h_box / 2) * scale_y

            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
            boxes_xywh = np.stack([x1, y1, w_box * scale_x, h_box * scale_y], axis=1)

            indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), filtered_scores.tolist(), conf_threshold, 0.45)
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                return postprocesar_deteccion(
                    boxes_xyxy[indices],
                    filtered_scores[indices],
                    filtered_class_ids[indices],
                    image,
                    threshold=conf_threshold,
                )
        return _to_rgb(image)
    except Exception as e:
        return _error_image(image, f"Error: {e}")


# -------------------------------------------------------------
# 5. Interfaz Gráfica con Gradio Blocks
# -------------------------------------------------------------
with gr.Blocks(title="Servidor de Inferencia ExecuTorch FP32") as demo:
    gr.Markdown(
        "# Servidor de Visión Artificial: ExecuTorch (Float32)\n"
        "Inferencia multimodelo optimizada con **ExecuTorch** y desplegada con **Gradio SDK** en Hugging Face Spaces."
    )

    with gr.Tab("Clasificación de Imágenes"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Imagen de entrada")
            label_out = gr.Label(num_top_classes=5, label="Predicciones Top-5")
        btn_run = gr.Button("Clasificar", variant="primary")
        btn_run.click(
            fn=predict_classification,
            inputs=img_in,
            outputs=label_out,
            api_name="predict_classification",
        )

    with gr.Tab("Segmentación Semántica"):
        with gr.Row():
            img_in_seg = gr.Image(type="pil", label="Imagen de entrada")
            img_out_seg = gr.Image(type="pil", label="Máscara de segmentación")
        btn_run_seg = gr.Button("Segmentar", variant="primary")
        btn_run_seg.click(
            fn=predict_segmentation,
            inputs=img_in_seg,
            outputs=img_out_seg,
            api_name="predict_segmentation",
        )

    with gr.Tab("Detección de Objetos"):
        with gr.Row():
            img_in_det = gr.Image(type="pil", label="Imagen de entrada")
            img_out_det = gr.Image(type="pil", label="Detecciones")
        btn_run_det = gr.Button("Detectar", variant="primary")
        btn_run_det.click(
            fn=predict_detection,
            inputs=img_in_det,
            outputs=img_out_det,
            api_name="predict_detection",
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
"""

with open(SPACE_DIR / "app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print(f"[+] Archivo app.py creado en {SPACE_DIR / 'app.py'}")

# %% [markdown]
# ## 3.2. Generación de Dependencias (`requirements.txt`)
#
# Para Hugging Face Spaces con el SDK nativo de Gradio definimos las dependencias en `requirements.txt`, permitiendo que la plataforma instale el entorno automáticamente sin necesidad de `Dockerfile`.

# %%
requirements_content = """torch>=2.1.0
torchvision
numpy
pillow
opencv-python-headless
gradio
executorch
"""

with open(SPACE_DIR / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements_content)

# Eliminar Dockerfile previo si existiera para evitar conflictos con el SDK de Gradio
if (SPACE_DIR / "Dockerfile").exists():
    (SPACE_DIR / "Dockerfile").unlink()

print(f"[+] Archivo requirements.txt creado en {SPACE_DIR / 'requirements.txt'}")

# %% [markdown]
# ## 3.3. Generación de los Metadatos `README.md` (Gradio SDK)
#
# Hugging Face requiere una cabecera YAML al inicio del `README.md`. Especificamos `sdk: gradio` junto con la versión instalada de Gradio (`gr.__version__`) para asegurar compatibilidad en el entorno remoto.

# %%
import gradio as gr

readme_code = f"""---
title: {SPACE_NAME}
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: {gr.__version__}
app_file: app.py
pinned: false
license: mit
---

# Servidor de Visión Artificial con ExecuTorch (Float32)
Despliegue interactivo de clasificación, segmentación y detección con **ExecuTorch** y **Gradio SDK**.
* **Clasificación:** MobileNetV2 (`mobilenet_v2.pte`)
* **Segmentación Semántica:** DeepLabV3 (`deeplabv3.pte`)
* **Detección de Objetos:** YOLO26 (`yolo26.pte`)
"""

with open(SPACE_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_code)

print(f"[+] Archivo README.md creado en {SPACE_DIR / 'README.md'} (sdk: gradio, sdk_version: {gr.__version__})")

# %% [markdown]
# ## 3.4. Copiado de los Modelos `.pte` al Directorio del Space
#
# Copiamos los artefactos `.pte` exportados en el cuaderno `Clase 12 - ExecuTorch` dentro de `SPACE_DIR`.

# %%
import shutil

for model_file in ["mobilenet_v2.pte", "deeplabv3.pte", "yolo26.pte"]:
    src_path = Path(model_file)
    if src_path.exists():
        shutil.copy(src_path, SPACE_DIR / model_file)
        size_mb = src_path.stat().st_size / (1024 * 1024)
        print(f"[+] Modelo copiado: {model_file} ({size_mb:.2f} MB)")
    else:
        print(f"[!] AVISO: No se encontró el modelo local '{model_file}'. Ejecuta primero 'Clase 12 - ExecuTorch'.")

print(f"\nContenido del directorio de despliegue '{SPACE_DIR}':")
for file in sorted(SPACE_DIR.iterdir()):
    print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")

# %% [markdown]
# # 4. Publicación y Despliegue en Hugging Face (`HfApi`)
#
# Subimos la carpeta completa directamente mediante `HfApi.create_repo` y `HfApi.upload_folder`. Este flujo crea el Space automáticamente si no existe y gestiona la carga de los archivos binarios `.pte` sin requerir `git clone`, `git lfs` ni `git push`.

# %%
from huggingface_hub import HfApi

api = HfApi()
repo_id = f"{HF_USER}/{SPACE_NAME}"

# 1. Crear el repositorio en Hugging Face si no existe (usando space_sdk="gradio")
try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    print(f"[+] Espacio '{repo_id}' confirmado en Hugging Face (SDK: gradio).")
except Exception as e:
    print(f"[!] Error o verificación de repositorio: {e}")

# 2. Subir todos los archivos generados y los modelos .pte
print("[*] Subiendo archivos y modelos al Space...")
api.upload_folder(
    folder_path=str(SPACE_DIR),
    repo_id=repo_id,
    repo_type="space",
    commit_message="Despliegue de modelos ExecuTorch FP32 con Gradio SDK"
)

print("\n[DESPLIEGUE INICIADO EXITOSAMENTE]")
print(f"Visita tu aplicación en vivo: https://huggingface.co/spaces/{repo_id}")

# %% [markdown]
# # 5. Pruebas de Cliente (Consumo Remoto de la API del Space)
#
# Una vez que Hugging Face termine de instalar las dependencias y construir el Space (usualmente toma entre 2 y 3 minutos en el primer despliegue), podemos consumir sus endpoints como una API desde Python usando `gradio_client`.
#
# ## 5.1. Consumo con `gradio_client`

# %%
from gradio_client import Client
try:
    from gradio_client import handle_file
except ImportError:
    handle_file = None

sample_img_path = "ruta/a/tu/imagen.jpg"  # <-- REEMPLAZA CON LA RUTA DE UNA IMAGEN LOCAL

if Path(sample_img_path).exists():
    try:
        # Conectar con el cliente de Gradio
        client = Client(f"{HF_USER}/{SPACE_NAME}")

        # Preparar payload compatible con versiones recientes de gradio_client
        img_payload = handle_file(sample_img_path) if handle_file is not None else sample_img_path

        # 1. Endpoint de clasificación
        result_cls = client.predict(img_payload, api_name="/predict_classification")
        print("[+] Clasificación de la imagen:", result_cls)

        # 2. Endpoint de segmentación semántica
        result_seg = client.predict(img_payload, api_name="/predict_segmentation")
        print("[+] Mapa de segmentación guardado en:", result_seg)

        # 3. Endpoint de detección de objetos
        result_det = client.predict(img_payload, api_name="/predict_detection")
        print("[+] Imagen con detecciones guardada en:", result_det)
    except Exception as e:
        print(f"[!] Detalle: {e}")
        print(f"Puedes verificar el log de ejecución de tu Space en: https://huggingface.co/spaces/{HF_USER}/{SPACE_NAME}")
else:
    print(f"[!] No se encontró la imagen de prueba '{sample_img_path}'. Actualiza la ruta para probar el cliente.")
