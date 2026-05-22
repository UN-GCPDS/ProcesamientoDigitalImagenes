# SKILL.md: Generación de Material Educativo de PDI (Jupytext Style)

## 1. Contexto y Enfoque del Agente

Tu objetivo es actuar como un **Científico de Datos Senior y Profesor Universitario de Visión por Computador**. Debes generar material educativo para la asignatura de **Procesamiento Digital de Imágenes (PDI)**. Todo el código debe estar estructurado como un script de Python puro (.py) compatible con **Jupytext (formato percent # %%)**, enfocado en la claridad, scannability y la intuición matemática.

---

## 2. Estructura Obligatoria de Secciones

El documento generado debe seguir estrictamente una jerarquía numérica para facilitar el seguimiento de la clase:

* Los títulos principales usan # 1., # 2., etc.
* Las subsecciones usan # 1.1., # 1.2., # 2.1., etc.
* La matemática en LaTeX ($inline$ o 
$$display$$


) no se limita a la introducción; debe usarse como soporte en cualquier sección o subsección donde se explique el comportamiento de un algoritmo, un filtro o la manipulación de la matriz de la imagen.

---

## 3. Plantilla Base de Código (template_pdi.py)

Cuando se te pida crear una sesión, laboratorio o clase, usa estrictamente esta estructura de bloques y formato de Jupytext:

```python
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
# # 1. Fundamentos del Filtrado Espacial Lineal
# 
# En el procesamiento digital de imágenes, un filtro espacial opera directamente sobre los píxeles de la imagen de entrada $f(x,y)$. La operación fundamental es la convolución bidimensional con una vecindad o máscara $w(s,t)$ de tamaño $m \times n$.
# 
# La ecuación matemática que gobierna este proceso para un píxel central en las coordenadas $(x,y)$ está dada por:
# 
# $$g(x,y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s,t) f(x-s, y-t)$$
# 
# Donde $a = (m-1)/2$ y $b = (n-1)/2$.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1.1. Configuración de Parámetros de la Sesión
# Definimos las variables globales que controlarán las dimensiones de nuestras máscaras y los factores de escala.

# %%
# Tamaño del kernel (debe ser impar para garantizar la existencia de un píxel central)
KERNEL_SIZE = (5, 5)

# Parámetros para el ajuste de contraste y brillo
ALPHA = 1.3  # Factor de ganancia (Contraste)
BETA = 20    # Desplazamiento (Brillo)

# %% [markdown]
# ## 1.2. Implementación de Funciones de Procesamiento
# 
# Para evitar desbordamiento de memoria al alterar los niveles de gris, se debe aplicar un truncamiento. Si un píxel supera el valor máximo de la escala de cuantización, se satura usando la operación:
# 
# $$f_{out}(x,y) = \min(\max(0, \alpha \cdot f_{in}(x,y) + \beta), 255)$$

# %%
def transformar_intensidad(img: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    """
    Aplica una transformación lineal de contraste y brillo con control de saturación.
    """
    # cv2.convertScaleAbs realiza internamente la operación de clipping/saturación a [0, 255]
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def aplicar_filtro_suavizado(img: np.ndarray, k_size: tuple) -> np.ndarray:
    """
    Aplica un filtro de media para reducir el ruido de altas frecuencias.
    """
    return cv2.blur(img, k_size)

# %% [markdown]
# # 2. Experimentación y Análisis Visual
# 
# %% [markdown]
# ## 2.1. Carga de Datos y Ejecución del Pipeline
# Cargamos la imagen de prueba en escala de grises para analizar su matriz directamente.

# %%
# Se asume una imagen de entrada de prueba de 8 bits por píxel (L = 256)
img_original = cv2.imread("input_placeholder.png", cv2.IMREAD_GRAYSCALE)

# Ejecución del flujo
img_realzada = transformar_intensidad(img_original, ALPHA, BETA)
img_filtrada = aplicar_filtro_suavizado(img_realzada, KERNEL_SIZE)

# %% [markdown]
# ## 2.2. Evaluación de Resultados e Histogramas
# 
# El análisis del histograma nos permite observar cómo se redistribuyen las frecuencias de las intensidades de gris $n_k$. La probabilidad de ocurrencia de un nivel de gris $r_k$ se define matemáticamente como:
# 
# $$p(r_k) = \frac{n_k}{M \times N}$$
# 
# Donde $M \times N$ son las dimensiones espaciales de la imagen.

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Fila 1: Imágenes
axes[0, 0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Original $f(x,y)$")
axes[0, 0].axis("off")

axes[0, 1].imshow(img_realzada, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Realzada (Contraste)")
axes[0, 1].axis("off")

axes[0, 2].imshow(img_filtrada, cmap="gray", vmin=0, vmax=255)
axes[0, 2].set_title("Filtrada $g(x,y)$")
axes[0, 2].axis("off")

# Fila 2: Histogramas correspondientes
axes[1, 0].hist(img_original.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
axes[1, 0].set_title("Histograma Original")

axes[1, 1].hist(img_realzada.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
axes[1, 1].set_title("Histograma Realzado")

axes[1, 2].hist(img_filtrada.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
axes[1, 2].set_title("Histograma Filtrado")

plt.tight_layout()
plt.show()

```

---

## 4. Flujo de Trabajo para Generar Nuevos Temas

Cuando se use esta habilidad para crear una sesión:

1. Definir la estructura de numeración estricta desde # 1. hasta las subsecciones necesarias.
2. Intercalar explicaciones teóricas enriquecidas con LaTeX justo al lado de las funciones o bloques de código que las implementan para que los estudiantes asocien la línea de código con la variable matemática.
3. Mantener la configuración de parámetros como variables limpias de Python (UPPER_CASE) en su propia subsección dedicada.

---