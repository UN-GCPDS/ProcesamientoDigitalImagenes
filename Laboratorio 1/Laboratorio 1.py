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
# # 1. Laboratorio 1: Detección Facial con OpenCV
#
# ## 1.1. Objetivo
#
# Construir un pipeline completo de visión por computador sobre rostros: detección con un modelo liviano preentrenado (YuNet, incluido en OpenCV 4 y 5) y análisis clásico del rostro recortado, aplicando los contenidos de las Clases 1 a 4: representación de la imagen como arreglo de NumPy, visualización, espacios de color con OpenCV, segmentación por umbralización en HSV, morfología matemática y detección de bordes. El flujo general es:
#
# $$foto \rightarrow YuNet \rightarrow ROI \rightarrow \begin{cases} gris \rightarrow ecualizado \rightarrow Canny \\ piel\ HSV \rightarrow morfología \end{cases} + ojos\ (landmarks)$$
#
# ## 1.2. Organización del trabajo
#
# * **Primera hora (práctica):** secciones 2 a 7 de este cuaderno, en orden y sin saltarse celdas. Tiempo total de cómputo inferior a 2 minutos en cualquier portátil.
# * **Segunda hora (informe):** completar dentro de este mismo cuaderno las tablas `[E1]`, `[E2]` y las respuestas `[P1]`, `[P2]`, `[P3]`. No se entrega ningún documento adicional.
# * **Entrega:** subir a Classroom el archivo `.ipynb` con todas las celdas ejecutadas (con salidas visibles) y las respuestas diligenciadas. La sección 8 verifica que la entrega esté completa antes de subirla.
#
# ## 1.3. Requisitos de software
#
# `Python 3.11`, `numpy`, `matplotlib` y `opencv-python` (versión 4.x o 5.x, ambas sirven):
#
# ```
# pip install opencv-python numpy matplotlib
# ```
#
# El detector usado (YuNet) funciona igual en OpenCV 4 y 5. La sección 2.1 verifica la instalación. En Google Colab no necesita instalar nada. El modelo de detección (`modelo/face_detection_yunet_2023mar.onnx`, 230 KB) viene en el repositorio; en Colab se descarga solo (sección 2.2).
#
# ## 1.4. Imágenes de trabajo
#
# Este laboratorio trabaja con exactamente **3 fotografías aportadas por el estudiante**, guardadas en la carpeta `img/` junto a este cuaderno, con estos nombres exactos:
#
# | Archivo | Contenido | Propósito |
# |---|---|---|
# | `frontal.jpg` | Usted de frente, buena iluminación, fondo claro, rostro descubierto (sin gafas de sol ni gorra) | Caso base: todo el pipeline debe funcionar aquí (se espera 1 cara) |
# | `grupo_3personas.jpg` | Usted con dos compañeros, los tres de frente a ~2 m, fondo despejado | Caso multiobjeto: conteo y efecto de los umbrales (se esperan 3 caras) |
# | `contraluz.jpg` | Un rostro frente a una ventana o fuente de luz (cara más oscura que el fondo) | Caso difícil: efecto de la ecualización y límites del detector |
#
# Tome las fotos antes de la práctica (celular en JPG, lado mayor ~800 px). Si alguna imagen falta, el cuaderno lo indica en la sección 2.3 y continúa con las disponibles, pero la entrega exige resultados sobre las 3.
#
# **Consentimiento y privacidad:** fotografíe únicamente a personas que acepten participar (incluido usted). Las imágenes se procesan solo en su equipo y el cuaderno resuelto se entrega únicamente por Classroom (uso institucional). No publique las fotos en redes ni las comparta fuera del curso.

# %% [markdown]
# ## 1.5. Datos del estudiante
#
# Diligencie la siguiente celda con sus datos. Es el primer campo que revisa la calificación.

# %%
ESTUDIANTE_NOMBRE = "Apellidos Nombres"  # <-- reemplazar
ESTUDIANTE_GRUPO = "Grupo"  # <-- reemplazar
FECHA = "2026-09-18"  # <-- reemplazar si es necesario

print(f"Estudiante: {ESTUDIANTE_NOMBRE} | Grupo: {ESTUDIANTE_GRUPO} | Fecha: {FECHA}")

# %% [markdown]
# # 2. Preparación del Entorno y de los Datos
#

# %% [markdown]
# ## 2.1. Verificación del entorno
#
# Ejecute la celda. Debe imprimir las versiones de OpenCV y NumPy y confirmar `Detector YuNet disponible: True` (vale tanto OpenCV 4 como 5). Si la verificación falla, deténgase y revise la instalación indicada en la sección 1.3 antes de continuar.

# %%
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Detector YuNet disponible:", hasattr(cv2, "FaceDetectorYN_create"))

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")

# Modelo de detección facial YuNet (liviano, ~230 KB). En local viene en el
# repositorio; en Colab se descarga automáticamente en la sección 2.2.
MODELO_NOMBRE = "face_detection_yunet_2023mar.onnx"
MODELO_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODELO_PATH = BASE_DIR / "modelo" / MODELO_NOMBRE

# Métricas del laboratorio. Cada ejercicio guarda aquí sus resultados;
# la sección 8 las resume para la entrega y la calificación.
METRICAS = {"nombre": ESTUDIANTE_NOMBRE, "grupo": ESTUDIANTE_GRUPO}

# Tamaño de figuras y ruta de salida (valores fijos, no se experimenta con ellos).
FIGSIZE = (15, 8)
SALIDA_PNG = BASE_DIR / "resultado.png"

# %% [markdown]
# ## 2.2. Carga de sus fotografías
#
# Coloque sus 3 fotos (sección 1.4) en la carpeta `img/` con los nombres exactos antes de continuar.
# * **En local:** copie los archivos a `img/` junto al cuaderno.
# * **En Google Colab:** ejecute la segunda celda de esta sección y seleccione sus 3 fotos cuando el navegador lo pida (deben llamarse `frontal.jpg`, `grupo_3personas.jpg` y `contraluz.jpg`; renómbrelas antes de subirlas).

# %%
import os

EN_COLAB = "COLAB_RELEASE_TAG" in os.environ
IMG_DIR = BASE_DIR / "img"
IMG_DIR.mkdir(exist_ok=True)
print("Entorno:", "Google Colab" if EN_COLAB else "local")
print("Carpeta img/:", IMG_DIR)

# %%
# Solo en Google Colab: suba sus 3 fotografías. En local esta celda no hace nada.
if EN_COLAB:
    try:
        from google.colab import files

        print("Seleccione sus 3 fotos (frontal.jpg, grupo_3personas.jpg, contraluz.jpg) ...")
        subidas = files.upload()
        for nombre, datos in subidas.items():
            destino = IMG_DIR / Path(nombre).name
            destino.write_bytes(datos)
            print(f"Guardada: {destino} ({len(datos) // 1024} KB)")
            # files.upload() deja además una copia en el directorio de trabajo; se elimina para no duplicar.
            duplicado = Path.cwd() / Path(nombre).name
            if duplicado.exists() and duplicado.resolve() != destino.resolve():
                duplicado.unlink()
    except ImportError:
        print("No se encontró el módulo de subida. Suba las fotos con el panel de archivos de Colab a img/ y vuelva a ejecutar.")
else:
    print("Celda solo para Google Colab. En local, copie sus fotos a img/ (ver sección 1.4).")

# %%
# Modelo de detección: en local ya viene en el repositorio; en Google Colab se
# descarga automáticamente (una vez, ~230 KB). En local esta celda solo verifica.
import urllib.request

if not MODELO_PATH.exists():
    if EN_COLAB:
        print("Descargando modelo YuNet ...")
        MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
    else:
        raise SystemExit(f"No se encuentra el modelo en {MODELO_PATH}. Verifique la carpeta del laboratorio.")
print(f"Modelo OK: {MODELO_PATH} ({MODELO_PATH.stat().st_size // 1024} KB)")

# %% [markdown]
# ## 2.3. Verificación de las imágenes
#
# Ejecute la celda. Para cada una de sus 3 fotos debe aparecer `OK` con sus dimensiones. Si aparece `[FALTA]`, revise el nombre del archivo y su ubicación en `img/` (sección 2.2): puede continuar con las disponibles, pero la entrega requiere las 3.

# %%
IMAGENES = ["frontal.jpg", "grupo_3personas.jpg", "contraluz.jpg"]

imagenes = {}
for nombre in IMAGENES:
    ruta = IMG_DIR / nombre
    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[FALTA] {nombre}: no está en img/. Coloque su foto con ese nombre (sección 2.2).")
    else:
        print(f"OK {nombre}: {img.shape[1]}x{img.shape[0]} px, dtype={img.dtype}")
        imagenes[nombre] = img

METRICAS["imagenes_encontradas"] = sorted(imagenes.keys())

if not imagenes:
    raise SystemExit("No hay imágenes en img/. No es posible continuar.")

fig, axes = plt.subplots(1, len(imagenes), figsize=(5 * len(imagenes), 5))
if len(imagenes) == 1:
    axes = [axes]
for ax, (nombre, bgr) in zip(axes, imagenes.items()):
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(nombre)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.4. Referencia de parámetros (lectura, 2 min)
#
# Cada ejercicio define sus parámetros al inicio de su propia celda: modifíquelos ahí mismo y vuelva a ejecutar la celda. No necesita subir hasta aquí ni tocar las funciones de la sección 3. Esta sección resume el fundamento:
#
# El detector YuNet asigna a cada candidato un puntaje de confianza $s$ y luego
# fusiona cajas traslapadas con supresión de no-máximos (NMS). Dos umbrales controlan el compromiso entre detecciones y falsos positivos (Ejercicio 2):
#
# $$detección = 1 \iff s \ge \tau_{score} \quad ; \quad IoU(A,B) = \frac{|A \cap B|}{|A \cup B|}, \; \text{se suprime si } IoU > \tau_{nms}$$
#
# * `SCORE_THRESHOLD` alto conserva solo caras seguras: menos falsos positivos, pero puede borrar caras difíciles.
# * `NMS_THRESHOLD` laxo (cercano a 1.0) deja sobrevivir cajas traslapadas duplicadas; estricto las fusiona (ver Clase 10: detectores NMS-Free).
#
# Un píxel se clasifica como piel si sus tres canales HSV están dentro del rango (Ejercicio 3):
# $$M(x,y) = 1 \iff H_{low} \le H \le H_{high} \land S_{low} \le S \le S_{high} \land V_{low} \le V \le V_{high}$$
# Rangos de OpenCV: H en 0-179, S y V en 0-255.
#
# Morfología: la apertura elimina puntos blancos aislados (sal) y el cierre
# rellena huecos negros (pimienta) (Ejercicio 3):
# $$A \circ B = (A \ominus B) \oplus B \quad ; \quad A \bullet B = (A \oplus B) \ominus B$$

# %% [markdown]
# # 3. Funciones de Procesamiento
#
# Estas funciones implementan el pipeline. Léalas para asociar cada operación con su fundamento (ecualización por acumulada $s_k = (L-1)\sum p(r_j)$, suavizado gaussiano $G(x,y)$, umbral de confianza, NMS por $IoU$, `inRange`, apertura/cierre). No requieren modificaciones.

# %%
def preprocesar(bgr: np.ndarray, ksize: tuple = (5, 5), sigma: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """Convierte BGR a gris, suaviza y ecualiza. Devuelve (gris, gris_ecualizado)."""
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, ksize, sigma)
    gris_eq = cv2.equalizeHist(gris)
    return gris, gris_eq


def crear_detector(score_thr: float, nms_thr: float, top_k: int) -> cv2.FaceDetectorYN:
    """Crea el detector YuNet con los umbrales dados."""
    return cv2.FaceDetectorYN_create(str(MODELO_PATH), "", (320, 320), score_thr, nms_thr, top_k)


def detectar_rostros(bgr: np.ndarray, score_thr: float, nms_thr: float, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detecta rostros en BGR. Devuelve (cajas Nx4 [x,y,w,h], landmarks Nx10, puntajes N).

    Cada fila de landmarks trae 5 puntos: ojo, ojo, nariz, boca, boca.
    """
    h, w = bgr.shape[:2]
    t0 = time.time()
    detector = crear_detector(score_thr, nms_thr, top_k)
    detector.setInputSize((w, h))
    _, caras = detector.detect(bgr)
    dt = time.time() - t0
    if caras is None:
        caras = np.zeros((0, 15), dtype=np.float32)
    cajas = caras[:, :4].astype(int)
    puntos = caras[:, 4:14]
    puntajes = caras[:, 14]
    print(f"  YuNet: {len(cajas)} cara(s) en {dt:.2f}s (score>={score_thr}, nms={nms_thr})")
    return cajas, puntos, puntajes


def segmentar_piel_hsv(bgr_roi: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Máscara binaria de piel en HSV mediante inRange."""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, low, high)


def limpiar_mascara(mask: np.ndarray, k_open: np.ndarray, k_close: np.ndarray) -> np.ndarray:
    """Apertura para quitar sal y cierre para tapar pimienta."""
    abierta = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    return cv2.morphologyEx(abierta, cv2.MORPH_CLOSE, k_close)


def dibujar(bgr: np.ndarray, cajas=(), puntos=None) -> np.ndarray:
    """Devuelve una copia anotada: verde = rostro, azul = ojos (landmarks de YuNet)."""
    out = bgr.copy()
    for (x, y, w, h) in cajas:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(out, "cara", (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if puntos is not None:
        for p in puntos:
            for k in (0, 1):  # los dos primeros landmarks son los ojos
                cv2.circle(out, (int(p[2 * k]), int(p[2 * k + 1])), 4, (255, 0, 0), -1)
    return out


# %% [markdown]
# # 4. Ejercicio 1 — Preproceso y Ecualización [10 min]
#
# **Procedimiento:**
# 1. Ejecute la celda y observe las tres columnas: imagen original, gris con suavizado y gris ecualizado, con sus histogramas debajo.
# 2. Compare el histograma del gris frente al ecualizado: la ecualización redistribuye las intensidades según la acumulada $s_k = (L-1)\sum_{j=0}^{k} p(r_j)$ con $p(r_j)=n_j/MN$.
# 3. Este ejercicio es formativo (sin entregable): su propósito es dominar histogramas y ecualización, el realce que el Ejercicio 4 aplica antes de detectar bordes en `contraluz.jpg`.

# %%
GAUSS_KSIZE = (5, 5)  # Tamaño del kernel gaussiano. Debe ser impar.
SIGMA = 0  # 0 = desviación calculada automáticamente desde el tamaño.

demo_nombre = "contraluz.jpg" if "contraluz.jpg" in imagenes else next(iter(imagenes))
demo = imagenes[demo_nombre]
gris_demo, gris_eq_demo = preprocesar(demo, GAUSS_KSIZE, SIGMA)

fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
axes[0, 0].imshow(cv2.cvtColor(demo, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"Original ({demo_nombre})")
axes[0, 0].axis("off")
axes[0, 1].imshow(gris_demo, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Gris + blur")
axes[0, 1].axis("off")
axes[0, 2].imshow(gris_eq_demo, cmap="gray", vmin=0, vmax=255)
axes[0, 2].set_title("Gris ecualizado")
axes[0, 2].axis("off")
axes[1, 0].hist(demo.ravel(), bins=128, range=[0, 256], color="gray")
axes[1, 0].set_title("Histograma BGR")
axes[1, 1].hist(gris_demo.ravel(), bins=256, range=[0, 256], color="black")
axes[1, 1].set_title("Histograma gris")
axes[1, 2].hist(gris_eq_demo.ravel(), bins=256, range=[0, 256], color="red")
axes[1, 2].set_title("Histograma ecualizado")
plt.tight_layout()
plt.show()

# %% [markdown]
# # 5. Ejercicio 2 — Detección de Rostros con YuNet [20 min]
#
# **Procedimiento:**
# 1. Ejecute la celda con los valores por defecto (`SCORE_THRESHOLD = 0.6`, `NMS_THRESHOLD = 0.3`). Registre en la Tabla `[E1]` el número de caras y los puntajes de cada una de sus 3 fotos. Si el conteo supera el número de personas visibles, no es un error: son falsos positivos; regístrelos tal cual y explíquelos en `[P1]`.
# 2. Experimento obligatorio A: cambie al inicio de esta misma celda `SCORE_THRESHOLD = 0.95`, vuelva a ejecutar y agregue la fila de `grupo_3personas.jpg`. Restaure 0.6.
# 3. Experimento obligatorio B: cambie al inicio de esta misma celda `NMS_THRESHOLD = 0.9`, vuelva a ejecutar y agregue la fila de `grupo_3personas.jpg`. Restaure 0.3.
# 4. Responda `[P1]` debajo de la tabla.

# %%
# Parámetros de este ejercicio (edítelos aquí para los experimentos A y B).
SCORE_THRESHOLD = 0.6  # Experimento A: comparar 0.6 frente a 0.95
NMS_THRESHOLD = 0.3  # Experimento B: comparar 0.3 frente a 0.9
TOP_K = 5000  # Máximo de candidatos antes de NMS.

resultados = {}

for nombre, bgr in imagenes.items():
    print(f"--- {nombre} ---")
    cajas, puntos, puntajes = detectar_rostros(bgr, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
    print(f"  puntajes: {sorted(round(float(s), 3) for s in puntajes)}")
    resultados[nombre] = {"n": len(cajas), "rostros": cajas, "puntos": puntos}
    anotada = dibujar(bgr, cajas, puntos)
    plt.figure(figsize=(6, 5))
    plt.imshow(cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB))
    plt.title(f"{nombre}: {len(cajas)} cara(s) [score>={SCORE_THRESHOLD}, nms={NMS_THRESHOLD}]")
    plt.axis("off")
    plt.show()

METRICAS["deteccion"] = {k: v["n"] for k, v in resultados.items()}
METRICAS["score_threshold"] = SCORE_THRESHOLD
METRICAS["nms_threshold"] = NMS_THRESHOLD

print("#caras (para transcribir a [E1])")
for nombre, r in resultados.items():
    print(f"{nombre} | {r['n']}")

# %% [markdown]
# ## Tabla [E1] — Resultados de detección (calificable)
#
# Transcriba aquí los conteos y puntajes impresos por la celda anterior. Las dos filas de `grupo_3personas.jpg` con `score=0.95` y `nms=0.9` corresponden a los experimentos obligatorios A y B.
#
# | imagen | #caras | puntajes | score_thr | nms_thr | observación (1 línea) |
# |---|---|---|---|---|---|
# | frontal.jpg |  |  | 0.6 | 0.3 |  |
# | grupo_3personas.jpg |  |  | 0.6 | 0.3 |  |
# | grupo_3personas.jpg |  |  | 0.95 | 0.3 |  |
# | grupo_3personas.jpg |  |  | 0.6 | 0.9 |  |
# | contraluz.jpg |  |  | 0.6 | 0.3 |  |
#
# *(Diligencie todas las celdas de la tabla. Sin esta tabla la entrega está incompleta.)*

# %% [markdown]
# ## Pregunta [P1] — Umbrales de confianza y NMS (calificable)
#
# Con los datos de la Tabla [E1]: ¿qué ocurrió al subir `score` a 0.95 (desaparecieron caras reales, falsos positivos, o nada)? ¿Y al relajar `nms` a 0.9 (aparecieron cajas duplicadas traslapadas)? Explique el mecanismo de cada umbral y su relación con los detectores modernos (Clase 10: NMS y detectores NMS-Free).
#
# *(Escriba su respuesta aquí. Mínimo 4 líneas con argumento técnico. No deje el marcador.)*

# %% [markdown]
# # 6. Ejercicio 3 — Piel en HSV y Limpieza Morfológica [15 min]
#
# **Procedimiento:**
# 1. Ejecute la celda: toma el rostro más grande de `frontal.jpg` (o de la primera imagen con detección), segmenta piel en HSV y compara la máscara cruda frente a la limpia con `OPEN 3x3 + CLOSE 5x5`.
# 2. Experimento obligatorio: cambie al inicio de esta misma celda `KERNEL_OPEN` a `(7,7)`, vuelva a ejecutar y registre ambos conteos de píxeles en la Tabla `[E2]`. Restaure `(3,3)`.
# 3. Si YuNet no detectó ningún rostro, la celda usa un recorte central para no interrumpir el laboratorio: indíquelo en `[E2]` como fallo del detector.
# 4. Responda `[P2]`.

# %%
# Parámetros de este ejercicio (edítelos aquí para el experimento).
# Rango de piel en HSV: H en 0-179, S y V en 0-255.
PIEL_LOW = np.array([0, 30, 60], dtype=np.uint8)
PIEL_HIGH = np.array([20, 150, 255], dtype=np.uint8)
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Experimento: comparar (3,3) frente a (7,7)
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

bgr_ref_nombre = (
    "frontal.jpg"
    if "frontal.jpg" in resultados and len(resultados["frontal.jpg"]["rostros"]) > 0
    else next((k for k, v in resultados.items() if len(v["rostros"]) > 0), None)
)
roi_desde_deteccion = True

if bgr_ref_nombre is not None:
    bgr_ref = imagenes[bgr_ref_nombre]
    rostros_ref = resultados[bgr_ref_nombre]["rostros"]
    x, y, w, h = max(rostros_ref, key=lambda r: r[2] * r[3])
    print(f"ROI de {bgr_ref_nombre}: x={x} y={y} w={w} h={h}")
else:
    roi_desde_deteccion = False
    bgr_ref_nombre = next(iter(imagenes))
    bgr_ref = imagenes[bgr_ref_nombre]
    H, W = bgr_ref.shape[:2]
    x, y, w, h = W // 4, H // 4, W // 2, H // 2
    print(f"YuNet no detectó rostros. Se usa un recorte central de {bgr_ref_nombre}.")

roi = bgr_ref[y : y + h, x : x + w]
mask_cruda = segmentar_piel_hsv(roi, PIEL_LOW, PIEL_HIGH)
mask_limpia = limpiar_mascara(mask_cruda, KERNEL_OPEN, KERNEL_CLOSE)
n_cruda, n_limpia = int(np.count_nonzero(mask_cruda)), int(np.count_nonzero(mask_limpia))
print(f"Piel: cruda={n_cruda} px, limpia={n_limpia} px")

METRICAS["piel"] = {
    "imagen": bgr_ref_nombre,
    "roi_desde_deteccion": roi_desde_deteccion,
    "cruda_px": n_cruda,
    "limpia_px": n_limpia,
    "kernel_open": KERNEL_OPEN.shape[:2],
}

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
axes[0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
axes[0].set_title("ROI rostro")
axes[0].axis("off")
axes[1].imshow(mask_cruda, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Máscara HSV cruda")
axes[1].axis("off")
axes[2].imshow(mask_limpia, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("OPEN + CLOSE")
axes[2].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Tabla [E2] — Máscara de piel (calificable)
#
# | kernel OPEN | máscara cruda (px) | máscara limpia (px) | ¿ROI desde detección o recorte central? |
# |---|---|---|---|
# | (3,3) |  |  |  |
# | (7,7) |  |  |  |
#
# *(Transcriba los conteos impresos por la celda para cada kernel.)*

# %% [markdown]
# ## Pregunta [P2] — Apertura y cierre (calificable)
#
# Compare ambas filas de la Tabla [E2]: ¿qué eliminó la apertura (ruido sal) y qué rellenó el cierre (pimienta)? ¿Qué ocurrió con el kernel de (7,7): limpió más o destruyó piel válida?
#
# *(Escriba su respuesta aquí. Mínimo 4 líneas con argumento técnico. No deje el marcador.)*

# %% [markdown]
# # 7. Ejercicio 4 — Rasgos, Bordes y Figura Final [15 min]
#
# **Procedimiento:**
# 1. Ejecute la primera celda: toma los ojos del rostro (landmarks que entrega YuNet, sin costo adicional) y compara el detector de bordes de Canny —aplicado sobre el gris ecualizado del Ejercicio 1— frente al gradiente morfológico.
# 2. El detector de Canny es multietapa (suavizado gaussiano → gradientes de Sobel → supresión de no-máximos → histéresis con `CANNY_T1/T2`):
#
# $$M = \sqrt{G_x^2+G_y^2}, \quad fuerte = M>T_2, \; débil = T_1<M\le T_2$$
#
# El gradiente morfológico no usa derivadas, solo geometría:
#
# $$G = (A \oplus B)-(A \ominus B)$$
#
# 3. Ejecute la segunda celda: genera el panel 2x3 y lo guarda como `resultado.png`. Esa imagen es el entregable `[FIG]`.
# 4. Responda `[P3]`.

# %%
# Parámetros de este ejercicio (umbrales de histéresis de Canny).
CANNY_T1 = 100  # Umbral bajo.
CANNY_T2 = 200  # Umbral alto. Experimento opcional: comparar (100,200) frente a (50,150).

gris_roi, gris_eq_roi = preprocesar(roi)

if roi_desde_deteccion:
    cajas_img = resultados[bgr_ref_nombre]["rostros"]
    puntos_img = resultados[bgr_ref_nombre]["puntos"]
    idx = int(np.argmax([w * h for (x, y, w, h) in cajas_img]))
    puntos_roi = puntos_img[idx].reshape(1, -1)
    n_ojos = 2
else:
    puntos_roi = np.zeros((0, 10), dtype=np.float32)
    n_ojos = 0
print(f"Ojos (landmarks YuNet): {n_ojos} (si el ROI es un recorte central no hay landmarks: regístrelo en [P3])")

canny = cv2.Canny(gris_eq_roi, CANNY_T1, CANNY_T2)
kernel_g = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
grad_morf = cv2.morphologyEx(gris_roi, cv2.MORPH_GRADIENT, kernel_g)
anotada_roi = dibujar(roi, [], puntos_roi)

METRICAS["rasgos"] = {"ojos": n_ojos}
METRICAS["canny"] = [CANNY_T1, CANNY_T2]

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
axes[0].imshow(cv2.cvtColor(anotada_roi, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Ojos detectados: {n_ojos}")
axes[0].axis("off")
axes[1].imshow(canny, cmap="gray")
axes[1].set_title(f"Canny ecualizado ({CANNY_T1},{CANNY_T2})")
axes[1].axis("off")
axes[2].imshow(grad_morf, cmap="gray")
axes[2].set_title("Gradiente morfológico")
axes[2].axis("off")
plt.tight_layout()
plt.show()

# %%
gris_full, gris_eq_full = preprocesar(bgr_ref)
canny_full = cv2.Canny(gris_eq_full, CANNY_T1, CANNY_T2)
n_caras_panel = len(resultados[bgr_ref_nombre]["rostros"]) if bgr_ref_nombre in resultados else 0
anotada_full = dibujar(
    bgr_ref,
    resultados[bgr_ref_nombre]["rostros"] if bgr_ref_nombre in resultados else [],
    resultados[bgr_ref_nombre]["puntos"] if bgr_ref_nombre in resultados else None,
)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(cv2.cvtColor(bgr_ref, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("1. Original")
axes[0, 0].axis("off")
axes[0, 1].imshow(gris_full, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("2. Gris")
axes[0, 1].axis("off")
axes[0, 2].imshow(cv2.cvtColor(anotada_full, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"3. YuNet: {n_caras_panel} cara(s)")
axes[0, 2].axis("off")
axes[1, 0].imshow(mask_limpia, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title("4. Piel HSV limpia")
axes[1, 0].axis("off")
axes[1, 1].imshow(cv2.cvtColor(anotada_roi, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("5. Ojos (landmarks)")
axes[1, 1].axis("off")
axes[1, 2].imshow(canny_full, cmap="gray")
axes[1, 2].set_title("6. Canny")
axes[1, 2].axis("off")
plt.tight_layout()
plt.savefig(SALIDA_PNG, dpi=150)
print(f"Figura [FIG] guardada en: {SALIDA_PNG}")
plt.show()

# %% [markdown]
# ## Pregunta [P3] — Límites del detector y del análisis por color (calificable)
#
# ¿En qué foto o condición el detector dudó o falló (contraluz, oclusión, perfil, fondo con textura)? ¿Por qué el espacio HSV separa la piel mejor que BGR? Con lo observado en el pipeline (detección + segmentación clásica), ¿qué problema seguiría sin resolver un detector mayor como YOLO (Unidad 2)?
#
# *(Escriba su respuesta aquí. Mínimo 4 líneas con argumento técnico. No deje el marcador.)*

# %% [markdown]
# # 8. Entrega en Classroom
#
# **Procedimiento:**
# 1. Ejecute la celda de verificación. Debe mostrar `ENTREGA LISTA`. Si indica faltantes, devuélvase a la sección señalada.
# 2. En el menú del cuaderno seleccione *Reiniciar kernel y ejecutar todo*, espere a que termine sin errores y guarde.
# 3. Suba a Classroom el archivo `.ipynb` de este cuaderno (con salidas visibles). No suba el `.py` ni las imágenes. En Google Colab, descárguelo primero con *Archivo > Descargar > Descargar .ipynb*.
#
# La calificación (humana o automática) revisa los identificadores `[E1]`, `[P1]`, `[E2]`, `[P2]`, `[P3]` y `[FIG]` dentro de este cuaderno, con los siguientes pesos:
#
# | ID | Contenido | Peso |
# |---|---|---|
# | Datos | `ESTUDIANTE_NOMBRE` y `GRUPO` diligenciados (sección 1.5) | 5% |
# | `[E1]` | Tabla de detección completa: sus 3 fotos + filas `score=0.95` y `nms=0.9` en el grupo | 25% |
# | `[P1]` | Efecto de los umbrales de confianza y NMS | 15% |
# | `[E2]` | Tabla de piel con ambos kernels | 15% |
# | `[P2]` | Apertura frente a cierre + efecto del kernel (7,7) | 15% |
# | `[P3]` | Límites del detector + HSV frente a BGR + puente a detectores mayores | 15% |
# | `[FIG]` | `resultado.png` generado y cuaderno ejecutado sin errores | 10% |
#
# Los conteos de `[E1]` deben coincidir con las salidas impresas por el cuaderno (cada estudiante tiene fotos distintas: se califica la coherencia, no un número fijo). Las respuestas `[P1]`–`[P3]` exigen argumento técnico; un marcador sin reemplazar vale cero.

# %%
import json

pendientes = []
if ESTUDIANTE_NOMBRE.strip().lower() in ("", "apellidos nombres"):
    pendientes.append("datos del estudiante (sección 1.5)")
if len(METRICAS.get("imagenes_encontradas", [])) < 3:
    pendientes.append(f"imágenes: se encontraron {METRICAS.get('imagenes_encontradas')}")
if "deteccion" not in METRICAS:
    pendientes.append("Ejercicio 2 sin ejecutar")
if "piel" not in METRICAS:
    pendientes.append("Ejercicio 3 sin ejecutar")
if "rasgos" not in METRICAS:
    pendientes.append("Ejercicio 4 sin ejecutar")
if not SALIDA_PNG.exists():
    pendientes.append("figura [FIG] (resultado.png no generado)")

print("=== RESUMEN DE ENTREGA (legible por máquina) ===")
print(json.dumps(METRICAS, indent=1, default=str))
print()
if pendientes:
    print("FALTA POR COMPLETAR:")
    for p in pendientes:
        print(f"  - {p}")
    print("Verifique además que las tablas [E1], [E2] y las respuestas [P1], [P2], [P3] estén diligenciadas.")
else:
    print("ENTREGA LISTA: verifique que [E1], [E2], [P1], [P2] y [P3] estén diligenciadas y suba el .ipynb a Classroom.")
