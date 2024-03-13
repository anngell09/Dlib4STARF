#CAPTURA INSTATANEA DE DESCRIPTORES, (Se cierra solo )



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
        forma = predictor(gray, cara)
        descriptores_faciales = np.array(facial_recognition_model.compute_face_descriptor(frame, forma))

        # Puedes imprimir o utilizar los descriptores faciales según tus necesidades
        print("Descriptores faciales capturados:", descriptores_faciales)

    # Mostrar el frame resultante
    cv2.imshow('Captura de Descriptores Faciales', frame)

    # Salir del bucle si se presiona 'q' o se han capturado los descriptores
    if cv2.waitKey(1) & 0xFF == ord('q') or descriptores_faciales is not None:
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
