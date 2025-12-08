import cv2
import numpy as np
import torch
import sys
import os

def preprocess_image(image_path, input_size=(224, 224)):
    """
    Carga una imagen, la convierte a RGB, la redimensiona y la prepara como tensor.
    """
    # Cargar imagen con OpenCV (BGR por defecto)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir de BGR a RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Redimensionar con OpenCV (usa NumPy internamente)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalizar a [0, 1] si los valores están en [0, 255]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Transponer de (H, W, C) a (C, H, W)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # Añadir batch dimension: (1, C, H, W)
    img_tensor = torch.from_numpy(img_transposed).unsqueeze(0)
    
    return img_tensor

def main(model_path, image_path, class_names=None, input_size=(224, 224)):
    # Verificar que los archivos existan
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    # Cargar el modelo TorchScript
    model = torch.jit.load(model_path)
    model.eval()  # modo evaluación

    # Preprocesar la imagen
    input_tensor = preprocess_image(image_path, input_size=input_size)

    # Inferencia
    with torch.no_grad():
        output_probs = model(input_tensor)  # ya son probabilidades

    # Asegurar que sea un tensor 1D
    probs = output_probs.squeeze().cpu().numpy()

    # Número de clases
    num_classes = len(probs)

    # Si no se dan nombres de clase, usar índices
    if class_names is None:
        class_names = [f"Clase_{i}" for i in range(num_classes)]
    elif len(class_names) != num_classes:
        print(f"Advertencia: número de nombres de clase ({len(class_names)}) no coincide con número de clases del modelo ({num_classes}). Usando índices.")
        class_names = [f"Clase_{i}" for i in range(num_classes)]

    # Imprimir resultados
    print("\n=== Resultados de la Inferencia ===")
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        print(f"{name}: {prob:.4f}")

    # Clase con mayor probabilidad
    pred_class_idx = np.argmax(probs)
    print(f"\nPredicción: {class_names[pred_class_idx]} (probabilidad: {probs[pred_class_idx]:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python inference_torchscript.py <ruta_modelo.pt> <ruta_imagen.jpg> [nombre_clase1 nombre_clase2 ...]")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    class_names = sys.argv[3:] if len(sys.argv) > 3 else None

    main(model_path, image_path, class_names=class_names)
