import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import time

# Cargar el modelo TFLite
interpreter = Interpreter(model_path="mobilenetv3.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dimensiones de entrada del modelo
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

# Iniciar captura de video
cap = cv2.VideoCapture(1)

print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocesar el frame
    img_resized = cv2.resize(frame, (width, height))
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Normalizar según el tipo de entrada del modelo
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(np.uint8)
    
    # Realizar inferencia y medir tiempo
    start_time = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    end_time = time.perf_counter()
    
    # Calcular tiempo de inferencia en milisegundos
    inference_time_ms = (end_time - start_time) * 1000
    
    # Obtener predicción
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data[0])
    confidence = output_data[0][prediction]
    
    # Mostrar resultado en el frame
    text = f"{prediction}: {confidence:.2%}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Mostrar tiempo de inferencia en milisegundos
    time_text = f"Inference: {inference_time_ms:.2f} ms"
    cv2.putText(frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Mostrar FPS basado en el tiempo de inferencia
    if inference_time_ms > 0:
        fps = 1000.0 / inference_time_ms
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 255), 2, cv2.LINE_AA)
    
    # Mostrar ventana
    cv2.imshow('Clasificacion en Tiempo Real', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()