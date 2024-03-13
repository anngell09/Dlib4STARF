#CAPTURA Y  COMPARACION EN VIVO FUNCIONAL v1.1



import cv2
import dlib
import numpy as np

# Cargar el modelo de predicción facial de dlib
predictor = dlib.shape_predictor("env/Lib/site-packages/dlib/models/shape_predictor_68_face_landmarks.dat")  # Asegúrate de descargar este modelo
facial_recognition_model = dlib.face_recognition_model_v1("env/Lib/site-packages/dlib/models/dlib_face_recognition_resnet_model_v1.dat")  # Asegúrate de descargar este modelo

# Inicializar el detector de caras de dlib
detector = dlib.get_frontal_face_detector()


# Función para extraer encodings de una imagen
def extraer_encodings(imagen): 
    # Cargar la imagen
    imagen = dlib.load_rgb_image(imagen)

    # Detectar caras en la imagen
    caras = dlib.get_frontal_face_detector()(imagen)

    # Si se detecta al menos una cara, extraer el encoding
    if caras:
        forma = predictor(imagen, caras[0])
        encoding = np.array(facial_recognition_model.compute_face_descriptor(imagen, forma))
        return encoding
    else:
        print("No se detectaron caras en la imagen.")
        return None

# Ruta de la imagen para la cual deseas extraer el encoding
imagen_a_procesar = "referencia.jpeg"

# Extraer el encoding
##Encodding de la imagen
encoding_imagen = extraer_encodings(imagen_a_procesar)



#####Exrtaccion de desdecriptores envivo



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
        if descriptores_faciales is None:
            forma = predictor(gray, cara)
            descriptores_faciales = np.array(facial_recognition_model.compute_face_descriptor(frame, forma))
            
            
            

    # Mostrar el frame resultante
    cv2.imshow('Captura de Descriptores Faciales', frame)
    
    
     # Calcular la distancia euclidiana entre los descriptores
    distancia = np.linalg.norm(encoding_imagen - descriptores_faciales)
    
      # Definir un umbral para decidir si las caras son lo suficientemente similares
    umbral = 0.6  # Ajusta según tus necesidades
    
    if distancia < umbral:
        print ("Las caras son similares")
    else:
        print("Las caras no son similares")
  

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
