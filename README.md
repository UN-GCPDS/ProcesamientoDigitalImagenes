# Procesamiento Digital de Imágenes

**Universidad Nacional de Colombia - Sede Manizales** · Facultad de Ingeniería y Arquitectura

Repositorio de la asignatura **Procesamiento Digital de Imágenes (PDI)**, que recorre desde los fundamentos de Python, NumPy, Matplotlib y OpenCV hasta técnicas modernas de aprendizaje profundo: clasificación, detección de objetos, segmentación y modelos generativos, finalizando con el despliegue de modelos en dispositivos edge y en la nube.

## 🌐 Sitio web

La página de la asignatura se publica con **GitHub Pages** desde la carpeta `Pagina/` y se despliega automáticamente mediante el workflow [`pages.yml`](.github/workflows/pages.yml) cada vez que se hace push a `main`.

- **URL del sitio:** `https://UN-GCPDS.github.io/ProcesamientoDigitalImagenes/`
- Estructura: página principal (`index.html`) + páginas por unidad (`unidad-1`, `unidad-2`, `unidad-3`) y página por clase (`clase-N.html`).
- Los logos institucionales viven en [`Pagina/assets/`](Pagina/assets/).

## 📚 Contenido

| Unidad | Clases | Temas |
|---|---|---|
| **1 · Procesamiento Digital de Imágenes** | 1–4 | Python y NumPy · Matplotlib · OpenCV, espacios de color y segmentación clásica · Transformaciones morfológicas y filtrado |
| **2 · Visión por Computador: Aprendizaje Profundo** | 5–11 | Perceptrón/MLP · CNNs y transfer learning · Segmentación y detección · SSD/SSDLite · Roboflow y YOLOv11 · NMS-Free · Modelos de difusión |
| **3 · Despliegue en nube y embebido** | 12 | ExecuTorch (inferencia edge) · Hugging Face Spaces y Gradio |

## 📁 Estructura del repositorio

```
├── .github/
│   └── workflows/pages.yml   # Deploy a GitHub Pages
├── Clase N/            # Material de cada clase (notebooks .ipynb, quizzes, figuras, PDFs)
├── Parcial 1/          # Evaluación parcial
├── Taller 1/           # Taller práctico
└── Pagina/             # Sitio web (GitHub Pages)
    ├── index.html
    ├── unidad-1/       # index.html + clase-1.html ... clase-4.html
    ├── unidad-2/       # index.html + clase-5.html ... clase-11.html
    ├── unidad-3/       # index.html + clase-12.html
    └── assets/         # logos institucionales
```

### Notas

- Cada carpeta `Clase N/` contiene el notebook (`.ipynb`), y en algunos casos el quiz (`Quizz.md`), figuras de ejemplo y la presentación (`.pdf`).
- Las carpetas `Clases raw/` (notebooks convertidos a `.py` con Jupytext y sinopsis) y `Pagina/mockup/` (mockups de diseño) son material de trabajo local y **no se incluyen** en el repositorio.

## 🛠️ Stack tecnológico

Python · NumPy · Matplotlib · OpenCV · PyTorch · TorchVision · Ultralytics (YOLO) · Roboflow · Hugging Face (diffusers, Spaces) · Gradio · ExecuTorch · scikit-learn · scipy

## 📄 Licencia

Este proyecto está licenciado bajo la [MIT License](LICENSE) (© 2026 UNAL-Manizales/GCPDS).
