# Procesamiento Digital de Imágenes

**Universidad Nacional de Colombia — Sede Manizales** · Facultad de Ingeniería y Arquitectura

> **Página oficial del curso:**
> ## [UN-GCPDS.github.io/ProcesamientoDigitalImagenes](https://UN-GCPDS.github.io/ProcesamientoDigitalImagenes/)
>
> El repositorio contiene el material del curso organizado por clases: notebooks, presentaciones y recursos de apoyo.

## Descripción del curso

La asignatura abarca desde los fundamentos del procesamiento digital de imágenes con Python, NumPy, Matplotlib y OpenCV, hasta técnicas de aprendizaje profundo para clasificación, detección de objetos, segmentación y modelos generativos. Concluye con el despliegue de modelos en dispositivos edge y en la nube.

Se requiere manejo básico de Python. No se requiere experiencia previa en visión por computador.

## Ruta de aprendizaje

El contenido se encuentra publicado en la [página del curso](https://UN-GCPDS.github.io/ProcesamientoDigitalImagenes/) y se organiza en tres unidades, las cuales se desarrollan en orden secuencial:

| Unidad | Contenidos | Clases |
|---|---|---|
| **1 · Procesamiento Digital de Imágenes** | Python y NumPy · Visualización con Matplotlib · OpenCV, espacios de color y segmentación clásica · Transformaciones morfológicas y filtrado | 1–4 |
| **2 · Visión por Computador: Aprendizaje Profundo** | Perceptrón y MLP · CNNs y transfer learning · Segmentación y detección · SSD y SSDLITE · Roboflow y YOLO · Detectores NMS-Free · Modelos de difusión | 5–11 |
| **3 · Despliegue en nube y embebido** | Exportación a edge con ExecuTorch · Despliegue en la nube con Hugging Face Spaces y Gradio · Portfolio y empaquetado final | 12 |

Cada página de clase incluye la fundamentación teórica, el notebook correspondiente y los enlaces al código fuente.

## Cursos de DataCamp (requisito de aprobación)

La aprobación de cada unidad requiere la culminación de los cursos de DataCamp asignados. Los enlaces se encuentran disponibles en la barra lateral de la página de cada unidad:

- **Unidad 1:** [Image Processing in Python](https://app.datacamp.com/learn/courses/image-processing-in-python) (4 capítulos, scikit-image y NumPy). Requisito para la aprobación de la Unidad 1.
- **Unidad 2:** [Deep Learning for Images with PyTorch](https://app.datacamp.com/learn/courses/deep-learning-for-images-with-pytorch) (4 capítulos: clasificación, detección, segmentación y GANs). Requisito para la aprobación de la Unidad 2.
- **Unidad 3:** no contempla curso de DataCamp. La unidad es de carácter práctico y se aprueba con el portfolio final.

## Evaluaciones

- **Taller 1** (Unidad 1): taller práctico de procesamiento clásico de imágenes.
- **Parcial 1** (Unidad 2): evaluación de visión por computador con aprendizaje profundo.
- **Portfolio final** (Unidad 3): empaquetado y publicación de la demostración final.

Las fechas y los criterios de entrega son anunciados por el docente en clase.

## Material y notebooks

Cada carpeta `Clase N/` contiene el notebook (`.ipynb`) de la clase y, según corresponda, el quiz (`Quizz.md`), las figuras de ejemplo y la presentación (`.pdf`).

El acceso al material se realiza desde la página de cada clase mediante el botón **Explorar Clase**. Los archivos `.ipynb` pueden ejecutarse en Google Colab o JupyterLab. El código fuente se encuentra disponible en este repositorio.

## Preguntas frecuentes

- **Acceso al contenido:** la totalidad del contenido se encuentra en la [página del curso](https://UN-GCPDS.github.io/ProcesamientoDigitalImagenes/). Se recomienda iniciar por la Unidad 1.
- **Organización del repositorio:** el material se encuentra distribuido en carpetas `Clase N/`, cada una con el contenido correspondiente a la sesión.
- **Requerimientos de hardware:** la Unidad 1 no requiere GPU. Para la Unidad 2 se recomienda el uso de Google Colab con aceleración por GPU.
- **Cursos de DataCamp:** son de carácter obligatorio y constituyen requisito de aprobación de las Unidades 1 y 2.

---

<details>
<summary><b>Información para colaboradores (despliegue)</b></summary>

El sitio web se encuentra en `Pagina/` (HTML estático con Tailwind mediante CDN) y se publica en GitHub Pages a través de [`.github/workflows/pages.yml`](.github/workflows/pages.yml) en cada push a `main`.

- Estructura: `index.html`, además de `unidad-1/`, `unidad-2/` y `unidad-3/` (cada una con `index.html` y sus archivos `clase-N.html`).
- Los archivos PDF de las presentaciones se copian durante la integración continua a `Pagina/assets/presentaciones/` (ver el workflow); por este motivo no se encuentran disponibles en el entorno local.
- El directorio `Pagina/mockup/` se excluye del artefacto publicado.

</details>

## Licencia

[MIT License](LICENSE) — © 2026 UNAL-Manizales / GCPDS.
