#Este no sirve

import cv2
import dlib
import numpy as np

# Cargar el modelo de predicción facial de dlib
predictor = dlib.shape_predictor("env/Lib/site-packages/dlib/models/shape_predictor_68_face_landmarks.dat")  # Asegúrate de descargar este modelo
facial_recognition_model = dlib.face_recognition_model_v1("env/Lib/site-packages/dlib/models/dlib_face_recognition_resnet_model_v1.dat")  # Asegúrate de descargar este modelo

# Inicializar el detector de caras de dlib
detector = dlib.get_frontal_face_detector()

# Inicializar la cámara (puedes ajustar el índice de la cámara según tu configuración)
cap = cv2.VideoCapture(0)

# Variable para almacenar los descriptores faciales
descriptores_faciales = None

# Variable para controlar si la comparación ya se realizó
comparacion_realizada = False

# Capturar descriptores faciales
while True:
    # Leer el frame desde la cámara
    ret, frame = cap.read()

    # Convertir a escala de grises para la detección facial
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el frame
    caras = detector(gray)

    # Iterar sobre las caras detectadas
    for cara in caras:
        # Obtener las coordenadas del rectángulo alrededor de la cara
        x, y, w, h = cara.left(), cara.top(), cara.width(), cara.height()

        # Dibujar un rectángulo alrededor de la cara
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extraer descriptores faciales
        if descriptores_faciales is None:
            forma = predictor(gray, cara)
            descriptores_faciales = np.array(facial_recognition_model.compute_face_descriptor(frame, forma))
            print("Descriptores faciales capturados:", descriptores_faciales)

            # Calcular la distancia euclidiana entre los descriptores
            encoding_imagen = [...]  # Sustituye [...] con los descriptores de la imagen de referencia
            distancia = np.linalg.norm(encoding_imagen - descriptores_faciales)

            # Definir un umbral para decidir si las caras son lo suficientemente similares
            umbral = 0.6  # Ajusta según tus necesidades

            # Imprimir el resultado de la comparación
            if distancia < umbral:
                print("Las caras son similares")
            else:
                print("Las caras no son similares")

            # Marcar que la comparación ya se realizó
            comparacion_realizada = True

    # Mostrar el frame resultante
    cv2.imshow('Captura de Descriptores Faciales', frame)

    # Salir del bucle si se presiona 'q' o si la comparación ya se realizó
    if cv2.waitKey(1) & 0xFF == ord('q') or comparacion_realizada:
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
