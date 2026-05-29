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
# # 1. Modelos de Difusión para Generación de Imágenes: Conceptos Matemáticos y Teóricos
# 
# Los **Modelos de Difusión** representan una de las familias de modelos generativos más exitosas de la actualidad en visión por computador y procesamiento digital de imágenes, superando a tecnologías previas como las GANs (Generative Adversarial Networks) y VAEs (Variational Autoencoders). 
# 
# El principio fundamental detrás de la difusión es aprender la distribución de datos destruyendo información de forma progresiva con ruido gaussiano y, posteriormente, entrenando una red neuronal para revertir este proceso de destrucción paso a paso.
# 
# ---
# 
# ## 1.1. El Proceso Directo (Forward Process / $q$)
# 
# El proceso hacia adelante (*Forward Process*) es un proceso de Markov que toma una imagen real de la distribución de datos, $x_0 \sim q(x)$, y le añade ruido gaussiano de manera sistemática y programada a lo largo de $T$ pasos de tiempo, siguiendo un programa de varianza (*variance schedule*) definido por $\beta_1, \beta_2, \dots, \beta_T$:
# 
# $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$
# 
# ### Muestreo Directo en un Paso Arbitrario
# Debido a las propiedades algebraicas de las variables gaussianas independientes, no es necesario calcular paso a paso la cadena de Markov para obtener $x_t$. Si definimos $\alpha_t = 1 - \beta_t$ y el producto acumulado $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, podemos escribir la relación condicional directa desde la imagen original $x_0$:
# 
# $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$
# 
# Esto nos permite expresar $x_t$ directamente como:
# 
# $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \quad \text{donde} \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
# 
# A medida que $t \to T$, el valor de $\bar{\alpha}_T \to 0$, por lo que la estructura de la imagen original se pierde por completo y $x_T$ se convierte en ruido blanco gaussiano puro.
# 
# ---
# 
# ## 1.2. El Proceso Inverso (Reverse Process / $p_\theta$)
# 
# El objetivo principal es generar una imagen nueva a partir de un ruido inicial aleatorio $x_T \sim \mathcal{N}(0, \mathbf{I})$ deshaciendo el ruido paso a paso. La transición inversa real $q(x_{t-1} | x_t)$ es intratable porque requiere integrar sobre todo el espacio de datos. Por lo tanto, aproximamos esta distribución de transición utilizando una red neuronal parametrizada por pesos $\theta$:
# 
# $$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$
# 
# ### Predicción del Ruido y Algoritmo de Muestreo (Sampling)
# En lugar de predecir la media del estado limpio $\mu_\theta$ directamente, es matemáticamente equivalente entrenar a la red neuronal (usualmente una **U-Net** con autoatención y embeddings de tiempo) para estimar el **ruido exacto $\epsilon$** que fue añadido al tensor en el paso $t$. Denotamos a este estimador como $\epsilon_\theta(x_t, t)$.
# 
# Una vez entrenado el modelo, el paso de eliminación de ruido para ir de $x_t$ a $x_{t-1}$ se define mediante la ecuación:
# 
# $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z \quad \text{donde} \quad z \sim \mathcal{N}(0, \mathbf{I})$$
# 
# Al repetir esta operación de manera iterativa desde $t = T$ hasta $t = 1$, el ruido se desvanece y emerge una estructura visual coherente.
# 
# ---
# 
# ## 1.3. Modelos de Difusión Latente (Latent Diffusion Models - LDM)
# 
# Aplicar el proceso de difusión directamente en el espacio de píxeles es computacionalmente prohibitivo. Por ejemplo, una imagen de $512 \times 512$ píxeles a color (3 canales) tiene un tamaño de dimensión de $786,432$ elementos. Entrenar una U-Net para predecir el ruido sobre tensores de este tamaño requiere una capacidad de cómputo enorme.
# 
# Los **Modelos de Difusión Latente (LDM)** solucionan este problema dividiendo el proceso en dos etapas principales:
# 
# 1. **Espacio Perceptual (VAE):** Se entrena un Autoencoder Variacional (VAE) compuesto por un codificador $\mathcal{E}$ y un decodificador $\mathcal{D}$. El codificador comprime una imagen $x$ a un espacio latente de baja dimensión $z = \mathcal{E}(x)$. Para una imagen de $512 \times 512 \times 3$, el tensor latente tiene un tamaño de $64 \times 64 \times 4$, lo cual reduce los datos espaciales en un factor de $8 \times 8 = 64$ veces (equivalente a $16,384$ elementos).
# 2. **Espacio de Difusión Latente:** El proceso de difusión directa (añadir ruido) y el proceso inverso (U-Net de eliminación de ruido) se ejecutan completamente dentro de este espacio latente compacto.
# 3. **Decodificación Final:** Una vez obtenido el latente limpio generado $z_0$, se pasa por el decodificador del VAE para reconstruir la imagen en el espacio de píxeles original:
# 
# $$\hat{x} = \mathcal{D}(z_0)$$
# 
# Gracias a este enfoque de espacio latente, modelos potentes de generación como **Stable Diffusion** pueden ser ejecutados de forma viable y rápida en hardware de consumo o en entornos de GPU con recursos moderados, como la GPU NVIDIA T4 de Google Colab (16 GB de VRAM).

# %%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

# %% [markdown]
# # 2. Configuración del Entorno y Parámetros de la Sesión
# 
# ## 2.1. Instalación de Dependencias
# Si estás ejecutando este script en Google Colab, primero debes instalar las librerías de Hugging Face dedicadas a la manipulación de modelos de difusión y procesamiento de lenguaje natural: `diffusers`, `transformers` y `accelerate`.
# 
# Descomenta y ejecuta la siguiente celda si estás en un entorno Colab o local sin estas dependencias instaladas.

# %%
# # Instalación de dependencias (Ejecutar solo si es necesario)
# !pip install diffusers transformers accelerate matplotlib -q

# %% [markdown]
# ## 2.2. Parámetros del Experimento y Configuración Global
# 
# Definimos las variables de control del flujo generativo. Para optimizar el uso de VRAM y garantizar una velocidad de generación óptima en la T4 de Colab, utilizaremos precisión de punto flotante de 16 bits (`torch.float16`) en GPU y cargaremos el modelo Stable Diffusion v1.5.

# %%
# Identificador del modelo en Hugging Face Hub
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Parámetros del prompt de generación (Entrada de texto)
PROMPT = "A detailed oil painting of an ancient library inside a giant hollow tree, cozy atmosphere, warm lighting, fantasy style, masterpiece"
NEGATIVE_PROMPT = "blurry, low quality, deformed, worst quality, distorted anatomy, text, watermarks"

# Parámetros de Inferencia del Difusor
NUM_INFERENCE_STEPS = 50   # Número de pasos de remoción de ruido (T)
GUIDANCE_SCALE = 7.5       # Control de adherencia al prompt (Classifier-Free Guidance)
SEED = 42                  # Semilla aleatoria para reproducibilidad

# Dimensiones de la imagen de salida (deben ser múltiplos de 8, preferiblemente 512 para SD v1.5)
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Detección y configuración del acelerador de hardware (GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"-> Entorno configurado en dispositivo: {DEVICE}")
if DEVICE == "cuda":
    print(f"-> Tarjeta gráfica activa: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# # 3. Pipeline de Generación con Hugging Face `diffusers`
# 
# La librería `diffusers` encapsula las interacciones complejas de los componentes de Stable Diffusion en un objeto unificado llamado `StableDiffusionPipeline`.
# 
# ## 3.1. Carga del Modelo Optimizado para Colab (FP16)
# 
# Cargamos el pipeline desde el repositorio oficial de Hugging Face. Especificamos `torch_dtype=torch.float16` y `use_safetensors=True` para acelerar la descarga y minimizar la ocupación en la GPU.

# %%
from diffusers import StableDiffusionPipeline

print("-> Cargando componentes del modelo Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    use_safetensors=True
)
pipe = pipe.to(DEVICE)
print("-> Modelo cargado y transferido al dispositivo correctamente.")

# %% [markdown]
# ### Exploración de los Submódulos del Pipeline
# El objeto `pipe` contiene los siguientes módulos fundamentales que interactúan en cada iteración:
# 
# *   **`pipe.tokenizer`**: Convierte las palabras del prompt en tokens (números de índice).
# *   **`pipe.text_encoder`**: Convierte los tokens en vectores continuos (embeddings contextuales) mediante una red CLIP de lenguaje.
# *   **`pipe.unet`**: Red neuronal convolucional residual con capas de atención que estima el ruido residual $\epsilon_\theta$ en el espacio latente.
# *   **`pipe.scheduler`**: Algoritmo matemático (por ejemplo, PNDM o DDIM) que calcula cómo remover el ruido estimado del paso actual para obtener el siguiente latente.
# *   **`pipe.vae`**: Autoencoder que mapea la imagen entre el espacio físico (píxeles) y el espacio comprimido (latentes).

# %% [markdown]
# ## 3.2. Inferencia Básica: Generación Completa en un Solo Paso
# 
# Ejecutamos el pipeline de forma tradicional pasando los parámetros de sesión. El método `__call__` del pipeline se encarga de tokenizar, codificar el texto, ejecutar el bucle de difusión y decodificar el latente resultante de forma automática.

# %%
# Fijar semilla aleatoria para garantizar reproducibilidad
generator = torch.Generator(device=DEVICE).manual_seed(SEED)

print("-> Iniciando proceso de generación básica...")
with torch.autocast(DEVICE):
    output = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        generator=generator
    )

# Extraemos la imagen resultante (PIL Image)
imagen_generada = output.images[0]

# Guardar la imagen localmente
imagen_generada.save("generacion_basica.png")

# Visualización
plt.figure(figsize=(8, 8))
plt.imshow(imagen_generada)
plt.title(f"Imagen Generada - Semilla {SEED}\nPrompt: '{PROMPT[:60]}...'")
plt.axis("off")
plt.show()

# %% [markdown]
# # 4. Inferencia Paso a Paso: Abriendo la Caja Negra del Bucle de Difusión
# 
# Para entender el flujo interno de los tensores de imagen y cómo interactúan las matemáticas de la U-Net y el Scheduler en el Procesamiento Digital de Imágenes, implementaremos el **bucle de inferencia de forma manual**.
# 
# ## 4.1. El Mecanismo de Guía Libre de Clasificador (Classifier-Free Guidance - CFG)
# 
# Para forzar al modelo a obedecer el prompt de texto, se calcula la predicción del ruido en dos condiciones simultáneamente:
# 
# 1.  **Predicción Condicionada ($\epsilon_c$):** El modelo recibe el embedding de nuestro prompt de texto.
# 2.  **Predicción No Condicionada ($\epsilon_\emptyset$):** El modelo recibe un prompt vacío o negativo (`unconditional input`), que actúa como la base de referencia.
# 
# Posteriormente, extrapolamos la predicción final de ruido $\epsilon_{\text{final}}$ alejándola de la versión no condicionada según la escala de guía $s$ (`GUIDANCE_SCALE`):
# 
# $$\epsilon_{\text{final}} = \epsilon_\emptyset + s \cdot (\epsilon_c - \epsilon_\emptyset)$$
# 
# Si $s = 1$, no hay guía adicional. Si $s > 1$ (típicamente $7.5$), el modelo exagera las características descritas en el prompt y reduce la variabilidad del fondo.

# %% [markdown]
# ## 4.2. Implementación del Bucle de Denoising Manual y Decodificación Temporal del VAE
# 
# Escribiremos el algoritmo de generación paso a paso, guardando y decodificando los tensores latentes intermedios con el VAE en intervalos específicos para registrar cómo se genera visualmente la imagen desde el ruido puro.

# %%
# Definir los pasos intermedios en los que decodificaremos el latente para ver el progreso visual
pasos_de_guardado = [0, 10, 20, 30, 40, NUM_INFERENCE_STEPS - 1]
imagenes_intermedias = {}

# 1. Codificación del texto a embeddings
# Tokenizar y codificar prompt positivo
text_inputs = pipe.tokenizer(
    PROMPT, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
)
with torch.no_grad():
    text_embeddings_cond = pipe.text_encoder(text_inputs.input_ids.to(DEVICE))[0]

# Tokenizar y codificar prompt negativo (para guía no condicionada)
uncond_inputs = pipe.tokenizer(
    NEGATIVE_PROMPT, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
)
with torch.no_grad():
    text_embeddings_uncond = pipe.text_encoder(uncond_inputs.input_ids.to(DEVICE))[0]

# Concatenamos los embeddings en un solo lote para procesarlos en paralelo por la U-Net
text_embeddings = torch.cat([text_embeddings_uncond, text_embeddings_cond])
text_embeddings = text_embeddings.to(pipe.unet.dtype) # Ajustar al tipo de datos de la red (float16/float32)

# 2. Inicialización del Tensor Latente con ruido gaussiano
# En SD v1.5, las dimensiones espaciales se reducen por un factor de 8 mediante el VAE
latents_shape = (1, pipe.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8)
generator = torch.Generator(device=DEVICE).manual_seed(SEED)

# Muestreo z_T ~ N(0, I)
latents = torch.randn(latents_shape, generator=generator, device=DEVICE, dtype=text_embeddings.dtype)

# 3. Preparación del Scheduler
pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)

# Escalar el ruido latente inicial según los valores esperados por el scheduler específico
latents = latents * pipe.scheduler.init_noise_sigma

print(f"Tensor latente inicial configurado. Shape: {latents.shape}")

# %%
# 4. Bucle principal de Denoising (Proceso inverso paso a paso)
for step, t in enumerate(tqdm(pipe.scheduler.timesteps, desc="Bucle de Difusión Manual")):
    
    # Duplicar los latentes para la inferencia paralela de CFG (condicionada + no condicionada)
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    
    # Predecir el ruido residual usando la U-Net
    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
    # Separar predicción condicionada y no condicionada
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    
    # Aplicar la fórmula de Classifier-Free Guidance (CFG)
    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
    
    # Calcular el estado latente anterior z_{t-1} a través del Scheduler
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decodificar y almacenar el latente en pasos específicos para ver la evolución
    if step in pasos_de_guardado:
        # Los latentes deben re-escalarse antes de pasar al VAE usando la constante de escala del modelo
        # 1 / 0.18215 es el factor de escala estándar definido en la arquitectura de Stable Diffusion
        latents_scaled = 1 / 0.18215 * latents
        
        with torch.no_grad():
            # Pasar por el Decodificador del VAE para llevar el latente al espacio de píxeles
            # Ecuación: x_reconstruido = D(z_t)
            image_decoded = pipe.vae.decode(latents_scaled).sample
            
        # Normalizar y recortar los valores de píxel al rango estándar [0.0, 1.0]
        image_decoded = (image_decoded / 2 + 0.5).clamp(0, 1)
        image_decoded = image_decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
        
        # Almacenar la imagen decodificada
        imagenes_intermedias[step] = image_decoded

# %% [markdown]
# ## 4.3. Visualización de la Evolución Temporal
# 
# Graficamos las imágenes reconstruidas de los pasos latentes intermedios. Observa cómo, a partir de una matriz caótica con valores distribuidos aleatoriamente, el modelo recupera progresivamente los bordes, texturas de alta frecuencia y finalmente los detalles cromáticos coherentes de la imagen.

# %%
fig, axes = plt.subplots(1, len(pasos_de_guardado), figsize=(20, 5))

for idx, step in enumerate(pasos_de_guardado):
    axes[idx].imshow(imagenes_intermedias[step])
    percent = int((step / (NUM_INFERENCE_STEPS - 1)) * 100)
    axes[idx].set_title(f"Paso {step} ({percent}%)")
    axes[idx].axis("off")

plt.suptitle("Evolución Temporal del Proceso de Difusión Inversa (Denoising en Espacio Latente)", fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig("evolucion_difusion.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# # 5. Modificaciones y Aplicaciones Avanzadas: Image-to-Image (Translación de Imagen)
# 
# La translación de imagen a imagen (*Image-to-Image / Img2Img*) es una técnica fundamental en el Procesamiento Digital de Imágenes que permite alterar el contenido de una imagen de entrada $x_0$ estructurándola bajo las directrices semánticas de un prompt de texto.
# 
# ## 5.1. Fundamentos Teóricos de Img2Img
# 
# En lugar de comenzar el proceso inverso desde ruido puro $z_T \sim \mathcal{N}(0, \mathbf{I})$, el pipeline realiza lo siguiente:
# 
# 1.  Codifica la imagen de entrada $x_0$ al espacio latente inicial $z_0 = \mathcal{E}(x_0)$.
# 2.  Añade ruido gaussiano de forma directa (Forward Process) hasta un paso de tiempo intermedio $t_{\text{start}}$ controlado por el parámetro de fuerza o intensidad del ruido $S \in [0, 1]$ (*Strength*):
# 
#     $$t_{\text{start}} = \text{int}(S \cdot T)$$
# 
#     $$z_{t_{\text{start}}} = \sqrt{\bar{\alpha}_{t_{\text{start}}}} z_0 + \sqrt{1 - \bar{\alpha}_{t_{\text{start}}}} \epsilon \quad \text{donde} \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
# 
# 3.  Ejecuta el proceso inverso de denoise (Reverse Process) únicamente desde $t_{\text{start}}$ hasta $0$ usando el prompt de texto guía.
# 
# ### El Impacto del Parámetro *Strength* ($S$)
# *   **Si $S \to 0$ ($t_{\text{start}} \to 0$):** Se añade muy poco ruido a la imagen inicial. La imagen de salida resultante es casi idéntica a la original y el prompt apenas tiene efecto.
# *   **Si $S \to 1$ ($t_{\text{start}} \to T$):** Se añade tanto ruido que la estructura de la imagen original colapsa por completo, comportándose como una generación de texto a imagen (*Text-to-Image*) convencional.

# %% [markdown]
# ## 5.2. Creación de una Imagen Base Simple
# Para asegurar la autonomía del script en cualquier entorno (incluyendo Colab) sin requerir descargas externas de archivos, crearemos una imagen sintética simple utilizando NumPy: un dibujo de un sol sobre montañas verdes con cielo degradado.

# %%
def crear_imagen_base() -> Image.Image:
    """ Genera una imagen sintética simple de 512x512 a color usando matrices de NumPy. """
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Fondo Degradado de cielo (Azul a Naranja)
    for y in range(h):
        r = int(255 * (y / h))
        g = int(120 * (y / h))
        b = int(200 + 55 * (1 - y / h))
        img[y, :, :] = [r, g, b]
        
    # 2. Agregar un sol circular (Círculo amarillo en la esquina superior derecha)
    cy, cx = 120, 380
    r_sol = 50
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    img[dist_from_center <= r_sol] = [255, 220, 0]
    
    # 3. Agregar montañas verdes (Parábolas inferiores)
    for x in range(w):
        y_montana = int(350 + 40 * np.sin(x / 40.0))
        img[y_montana:, x, :] = [34, 139, 34] # Verde bosque
        
    return Image.fromarray(img)

# Generar y mostrar la imagen de entrada base
imagen_base = crear_imagen_base()
imagen_base.save("imagen_base.png")

plt.figure(figsize=(5, 5))
plt.imshow(imagen_base)
plt.title("Imagen Sintética de Entrada ($x_0$)")
plt.axis("off")
plt.show()

# %% [markdown]
# ## 5.3. Carga del Pipeline de Image-to-Image y Generación Comparativa
# 
# Cargamos el pipeline de traducción `StableDiffusionImg2ImgPipeline` compartiendo los pesos de los componentes que ya descargamos previamente para optimizar la memoria RAM y VRAM.

# %%
from diffusers import StableDiffusionImg2ImgPipeline

print("-> Inicializando el pipeline de Image-to-Image...")
# Cargamos reutilizando los componentes descargados en el pipeline de texto a imagen
pipe_img2img = StableDiffusionImg2ImgPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=pipe.unet,
    scheduler=pipe.scheduler,
    safety_checker=pipe.safety_checker,
    feature_extractor=pipe.feature_extractor
)
pipe_img2img = pipe_img2img.to(DEVICE)

# %% [markdown]
# Evaluaremos el comportamiento del modelo utilizando un prompt fantástico para transformar nuestro paisaje básico, probando diferentes niveles de fuerza de ruido $S \in \{0.25, 0.55, 0.85\}$.

# %%
# Parámetros para la translación de imagen
PROMPT_IMG2IMG = "A breathtaking painting of a futuristic castle on top of high cliffs, waterfalls, aurora borealis in the sky, hyperdetailed digital art"
valores_strength = [0.25, 0.55, 0.85]
imagenes_resultados = []

# Pre-procesar la imagen base convirtiéndola a las dimensiones de inferencia
imagen_base_resized = imagen_base.resize((IMG_WIDTH, IMG_HEIGHT))

# Ejecutar inferencia para cada nivel de fuerza
for strength in valores_strength:
    print(f"-> Generando traducción de imagen con Strength = {strength}...")
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    
    with torch.autocast(DEVICE):
        output = pipe_img2img(
            prompt=PROMPT_IMG2IMG,
            negative_prompt=NEGATIVE_PROMPT,
            image=imagen_base_resized,
            strength=strength,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator
        )
    
    imagenes_resultados.append(output.images[0])

# %% [markdown]
# ## 5.4. Visualización de Resultados Comparativos
# 
# Graficamos el resultado de las traducciones comparándolas directamente con la imagen sintética original de entrada.

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Imagen original de entrada
axes[0].imshow(imagen_base_resized)
axes[0].set_title("Imagen Original ($x_0$)\n(Sintética de NumPy)")
axes[0].axis("off")

# Resultados con distintos strengths
for i, strength in enumerate(valores_strength):
    axes[i + 1].imshow(imagenes_resultados[i])
    axes[i + 1].set_title(f"Resultados con S = {strength}\n(Prompt: '{PROMPT_IMG2IMG[:25]}...')")
    axes[i + 1].axis("off")
    
    # Guardar las imágenes individualmente
    imagenes_resultados[i].save(f"resultado_strength_{strength}.png")

plt.suptitle("Impacto del Parámetro 'Strength' ($S$) en la Traducción de Imagen a Imagen", fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig("comparacion_img2img.png", bbox_inches='tight')
plt.show()
