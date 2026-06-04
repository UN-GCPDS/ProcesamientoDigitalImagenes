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
# # 1. Servir Modelos de ExecuTorch en Hugging Face Spaces (Gradio)
#
# Una vez que los modelos han sido exportados al formato optimizado de ExecuTorch (`.pte`), el siguiente paso es desplegarlos en producción para que puedan ser consumidos por aplicaciones externas. Hugging Face Spaces proporciona un entorno ideal para hospedar aplicaciones web interactivas de aprendizaje automático utilizando la librería **Gradio**.
#
# ## 1.1. Arquitectura de Despliegue en Hugging Face
#
# En este laboratorio, construiremos un servidor Gradio multimodelo que puede procesar tareas de:
# 1.  **Clasificación de imágenes**
# 2.  **Segmentación semántica**
# 3.  **Detección de objetos**
#
# Debido a que ExecuTorch requiere librerías nativas compiladas de C++ que pueden variar entre sistemas operativos, implementaremos un cargador robusto con un **mecanismo de fallback (contingencia)**. Si el entorno no dispone del runtime nativo de ExecuTorch (`executorch.runtime`), la aplicación cargará automáticamente los modelos base de PyTorch en FP32/FP16 para garantizar que la interfaz de usuario siga funcionando correctamente.
#

# %%
import os
import sys
import shutil
import base64
import requests
from PIL import Image
from huggingface_hub import notebook_login

# %% [markdown]
# # 2. Creación y Configuración del Repositorio de Hugging Face
#
# Para automatizar la subida y configuración del Space, primero debemos autenticarnos con Hugging Face. Esto abrirá una interfaz interactiva donde ingresarás tu **User Access Token** (con permisos de *write*).

# %%
try:
    notebook_login()
except Exception as e:
    print("[INFO] notebook_login no está disponible o no se ejecuta en un entorno interactivo. Continúa con la generación de archivos locales.")

# %% [markdown]
# ## 2.1. Configuración de Variables del Servidor
#
# Define tu nombre de usuario de Hugging Face y el nombre que deseas darle a tu nuevo Space.

# %%
HF_USER = "tu_usuario_hf"
SPACE_NAME = "executorch-pdi-unal"
REPO_URL = f"https://huggingface.co/spaces/{HF_USER}/{SPACE_NAME}"

print(f"URL de clonación del Space: {REPO_URL}")

# %% [markdown]
# ## 2.2. Clonación y Configuración de Git LFS
#
# Clonamos el repositorio del Space. Dado que subiremos archivos de modelos pesados (`.pte`), debemos configurar Git LFS (Large File Storage) para evitar rechazos en el comando `git push`.

# %%
# 1. Clonar el repositorio del space creado
# !git clone {REPO_URL}

# %%
# 2. Configurar Git LFS para que gestione los modelos binarios .pte
# %cd {SPACE_NAME}
# !git lfs track "*.pte"
# !git add .gitattributes
# %cd ..

# %% [markdown]
# # 3. Generación de los Archivos de Despliegue (Gradio & Docker)
#
# Escribiremos dinámicamente los tres archivos esenciales dentro del directorio de nuestro Space:
# 1.  `app.py`: El código de la aplicación de inferencia Gradio.
# 2.  `Dockerfile`: La definición de dependencias del contenedor de Docker.
# 3.  `README.md`: Los metadatos de configuración que Hugging Face requiere para entender el SDK y compilar el contenedor.

# %%
# 1. Generar app.py con la aplicación Gradio en Float32
app_content = """import os
import sys
import numpy as np
import cv2
import torch
import torchvision.models as models
import gradio as gr
from PIL import Image

try:
    from executorch.runtime import Runtime, Program
    EXECUTORCH_AVAILABLE = True
    print("[INFO] Runtime de ExecuTorch cargado correctamente.")
except ImportError:
    EXECUTORCH_AVAILABLE = False
    print("[WARNING] ExecuTorch no está disponible. Usando fallback de PyTorch.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] Ultralytics no está disponible.")

PATH_MODEL_CLS = "mobilenet_v2.pte"
PATH_MODEL_SEG = "deeplabv3.pte"
PATH_MODEL_DET = "yolo26.pte"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

np.random.seed(42)
COLOR_PALETTE = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

def preprocesar_imagen(img_pil: Image.Image, size: tuple, normalizar: bool = True) -> torch.Tensor:
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    img_resized = img_pil.resize(size)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    if normalizar:
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_transposed = np.transpose(img_np, (2, 0, 1))
    tensor = torch.from_numpy(img_transposed).unsqueeze(0).contiguous()
    return tensor

def postprocesar_segmentacion(output_tensor: torch.Tensor, original_img: Image.Image) -> Image.Image:
    mask = torch.argmax(output_tensor[0], dim=0).numpy().astype(np.uint8)
    w, h = original_img.size
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_color = COLOR_PALETTE[mask_resized]
    img_orig_np = np.array(original_img)
    blended = cv2.addWeighted(img_orig_np, 0.6, mask_color, 0.4, 0)
    return Image.fromarray(blended)

def postprocesar_deteccion(boxes, scores, labels, original_img: Image.Image, threshold: float = 0.25) -> Image.Image:
    img_np = np.array(original_img)
    h, w, _ = img_np.shape
    
    boxes_np = boxes.detach().numpy() if isinstance(boxes, torch.Tensor) else np.array(boxes)
    scores_np = scores.detach().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
    labels_np = labels.detach().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    
    for box, score, label_idx in zip(boxes_np, scores_np, labels_np):
        if score >= threshold:
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(0, min(xmax, w - 1))
            ymax = max(0, min(ymax, h - 1))
            
            label_text = f"{COCO_CLASSES[int(label_idx) % len(COCO_CLASSES)]}: {score:.2f}"
            color = [int(c) for c in COLOR_PALETTE[int(label_idx) % len(COLOR_PALETTE)]]
            
            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), color, 3)
            cv2.putText(img_np, label_text, (xmin, max(ymin - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return Image.fromarray(img_np)

class ModelRunner:
    def __init__(self, pte_path: str, fallback_model_fn):
        self.use_executorch = EXECUTORCH_AVAILABLE and os.path.exists(pte_path)
        
        if self.use_executorch:
            print(f"[INFO] Cargando modelo ExecuTorch: {pte_path}")
            try:
                self.runtime = Runtime.get()
                self.program = self.runtime.load_program(pte_path)
                self.method = self.program.load_method("forward")
            except Exception as e:
                print(f"[ERROR] Error al cargar modelo ExecuTorch {pte_path}: {e}. Usando fallback.")
                self.use_executorch = False
                
        if not self.use_executorch:
            print(f"[INFO] Cargando fallback de PyTorch para: {pte_path}")
            try:
                self.model = fallback_model_fn()
                if hasattr(self.model, "eval"):
                    try:
                        self.model = self.model.eval()
                    except Exception:
                        pass
                if hasattr(self.model, "model") and hasattr(self.model.model, "eval"):
                    try:
                        self.model.model.eval()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[ERROR] Error al cargar fallback: {e}")
                self.model = None

    def run(self, input_tensor: torch.Tensor):
        if self.use_executorch:
            outputs = self.method.execute((input_tensor,))
            if isinstance(outputs, list) and len(outputs) == 1:
                return outputs[0]
            return outputs
        else:
            with torch.no_grad():
                return self.model(input_tensor)

runner_cls = ModelRunner(PATH_MODEL_CLS, lambda: models.mobilenet_v2(pretrained=True))
runner_seg = ModelRunner(PATH_MODEL_SEG, lambda: models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True))
runner_det = ModelRunner(PATH_MODEL_DET, lambda: YOLO("yolo26n.pt") if YOLO_AVAILABLE else None)

import urllib.request
try:
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    imagenet_classes = urllib.request.urlopen(url).read().decode('utf-8').splitlines()
except Exception:
    imagenet_classes = [f"Clase {i}" for i in range(1000)]

def predict_classification(image: Image.Image) -> dict:
    if image is None:
        return {}
    try:
        if not (runner_cls.use_executorch or (hasattr(runner_cls, "model") and runner_cls.model is not None)):
            return {"Error": 1.0, "Modelo de clasificacion no cargado": 0.0}
        tensor = preprocesar_imagen(image, (224, 224))
        output = runner_cls.run(tensor)
        if isinstance(output, list):
            output = output[0]
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)
            
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 5)
        return {imagenet_classes[int(idx)]: float(prob) for prob, idx in zip(top_prob, top_catid)}
    except Exception as e:
        return {"Error": 1.0, str(e): 0.0}

def predict_segmentation(image: Image.Image) -> Image.Image:
    if image is None:
        return None
    try:
        if not (runner_seg.use_executorch or (hasattr(runner_seg, "model") and runner_seg.model is not None)):
            img_np = np.array(image)
            cv2.putText(img_np, "Modelo de segmentacion no cargado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return Image.fromarray(img_np)
        tensor = preprocesar_imagen(image, (256, 256))
        output = runner_seg.run(tensor)
        if isinstance(output, dict):
            output_tensor = output["out"]
        else:
            output_tensor = output
        if isinstance(output_tensor, np.ndarray):
            output_tensor = torch.from_numpy(output_tensor)
        return postprocesar_segmentacion(output_tensor, image)
    except Exception as e:
        img_np = np.array(image)
        cv2.putText(img_np, f"Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return Image.fromarray(img_np)

def predict_detection(image: Image.Image) -> Image.Image:
    if image is None:
        return None
    try:
        if not (runner_det.use_executorch or (hasattr(runner_det, "model") and runner_det.model is not None)):
            img_np = np.array(image)
            cv2.putText(img_np, "Modelo de deteccion no cargado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return Image.fromarray(img_np)
            
        if runner_det.use_executorch:
            tensor = preprocesar_imagen(image, (640, 640), normalizar=False)
            output = runner_det.run(tensor)
            pred = output[0].numpy() if isinstance(output, torch.Tensor) else output[0]
            predictions = pred[0].T if len(pred.shape) == 3 else pred.T
            
            boxes = predictions[:, :4]
            scores = predictions[:, 4:]
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            conf_threshold = 0.25
            keep = max_scores >= conf_threshold
            filtered_boxes = boxes[keep]
            filtered_scores = max_scores[keep]
            filtered_class_ids = class_ids[keep]
            
            if len(filtered_boxes) > 0:
                cx, cy, w_box, h_box = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
                orig_h, orig_w = image.size[1], image.size[0]
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0
                
                x1 = (cx - w_box / 2) * scale_x
                y1 = (cy - h_box / 2) * scale_y
                x2 = (cx + w_box / 2) * scale_x
                y2 = (cy + h_box / 2) * scale_y
                
                boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
                boxes_xywh = np.stack([x1, y1, w_box * scale_x, h_box * scale_y], axis=1)
                
                indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), filtered_scores.tolist(), conf_threshold, 0.45)
                if len(indices) > 0:
                    indices = np.array(indices).flatten()
                    return postprocesar_deteccion(boxes_xyxy[indices], filtered_scores[indices], filtered_class_ids[indices], image, threshold=conf_threshold)
            return image
        else:
            results = runner_det.model(image, verbose=False)
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy()
            return postprocesar_deteccion(boxes, scores, labels, image, threshold=0.25)
    except Exception as e:
        img_np = np.array(image)
        cv2.putText(img_np, f"Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return Image.fromarray(img_np)

with gr.Blocks(title="Servidor de Inferencia ExecuTorch FP32") as demo:
    gr.Markdown("# Servidor de Visión Artificial: ExecuTorch (Float32)")
    gr.Markdown("Inferencia multimodelo optimizada con ExecuTorch y desplegada con Docker en Hugging Face Spaces.")
    
    with gr.Tab("Clasificación de Imágenes"):
        with gr.Row():
            img_in = gr.Image(type="pil")
            label_out = gr.Label(num_top_classes=5)
        btn_run = gr.Button("Clasificar")
        btn_run.click(predict_classification, inputs=img_in, outputs=label_out)
        
    with gr.Tab("Segmentación Semántica"):
        with gr.Row():
            img_in_seg = gr.Image(type="pil")
            img_out_seg = gr.Image(type="pil")
        btn_run_seg = gr.Button("Segmentar")
        btn_run_seg.click(predict_segmentation, inputs=img_in_seg, outputs=img_out_seg)
        
    with gr.Tab("Detección de Objetos"):
        with gr.Row():
            img_in_det = gr.Image(type="pil")
            img_out_det = gr.Image(type="pil")
        btn_run_det = gr.Button("Detectar")
        btn_run_det.click(predict_detection, inputs=img_in_det, outputs=img_out_det)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
"""

os.makedirs(SPACE_NAME, exist_ok=True)
with open(f"{SPACE_NAME}/app.py", "w", encoding="utf-8") as f:
    f.write(app_content)
print("app.py creado con éxito en el space.")

# %%
# 2. Generar Dockerfile con el entorno de compilación de ExecuTorch
dockerfile_content = """FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --quiet \
    executorch \
    torch==2.11.0 \
    torchvision \
    numpy \
    opencv-python-headless \
    gradio \
    gradio_client \
    ultralytics

COPY --chown=user app.py ./app.py
COPY --chown=user *.pte ./

EXPOSE 7860

CMD ["python", "app.py"]
"""

with open(f"{SPACE_NAME}/Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_content)
print("Dockerfile creado con éxito en el space.")

# %%
# 3. Generar README.md con metadatos YAML correctos para Hugging Face Spaces
readme_content = f"""---
title: {SPACE_NAME}
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Inferencia de Visión con ExecuTorch (Float32)
Despliegue interactivo de clasificación, segmentación y detección con Gradio y Docker.
"""

with open(f"{SPACE_NAME}/README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("README.md creado con éxito en el space.")

# %% [markdown]
# ## 3.1. Copiado de los Modelos `.pte`
#
# Copiamos los archivos de los modelos exportados dentro de la carpeta del repositorio local del Space.

# %%
# Copiar modelos exportados al directorio del space
for model_file in ["mobilenet_v2.pte", "deeplabv3.pte", "yolo26.pte"]:
    if os.path.exists(model_file):
        shutil.copy(model_file, f"{SPACE_NAME}/")
        print(f"Modelo copiado: {model_file}")
    else:
        print(f"[WARNING] No se encontró el modelo local '{model_file}'. Recuerda exportarlo primero.")

# %% [markdown]
# # 4. Publicación y Despliegue en el Space
#
# Configuramos el nombre de usuario de Git, realizamos el commit con todos los archivos agregados y enviamos los cambios para activar la compilación del contenedor Docker en Hugging Face.

# %%
# %cd {SPACE_NAME}
# !git config --global user.email "tu_correo@unal.edu.co"
# !git config --global user.name "tu_usuario_hf"
# !git add .
# !git commit -m "Despliegue inicial de modelos ExecuTorch FP32 en Hugging Face Spaces"
# !git push
# %cd ..

# %% [markdown]
# # 5. Pruebas de Cliente (Consumo del API del Space)
#
# Cuando el contenedor finalice de compilarse en Hugging Face, la API web estará expuesta. Podemos enviar solicitudes REST desde Python.

# %% [markdown]
# ## 5.1. `gradio_client`
#
# Método recomendado para interactuar directamente con la API desde Python.

# %%
try:
    from gradio_client import Client
except ImportError:
    print("[INFO] gradio_client no está instalado localmente. Omitiendo prueba de cliente de Gradio.")

# URL de tu Space desplegado (Público o Privado con Token)
# client = Client(f"{HF_USER}/{SPACE_NAME}", hf_token="tu_token_opcional")
# result = client.predict(image="ruta/a/tu/imagen.jpg", api_name="/predict_classification")
# print("Clasificación de la imagen:", result)
