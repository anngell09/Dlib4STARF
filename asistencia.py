#CAPTURA Y  COMPARACION EN VIVO FUNCIONAL



import cv2
import dlib
import numpy as np

#  Inicializar los valores de los modelos
predictor = dlib.shape_predictor("env/Lib/site-packages/dlib/models/shape_predictor_68_face_landmarks.dat")  
facial_recognition_model = dlib.face_recognition_model_v1("env/Lib/site-packages/dlib/models/dlib_face_recognition_resnet_model_v1.dat")  

# Inicializar el detector de caras de dlib
detector = dlib.get_frontal_face_detector()


# funcion para  extraer encodings de una imagen <<<<<<<<<<<
def extraer_encodings(imagen):
    
    # valor de la imagen
    imagen = dlib.load_rgb_image(imagen)

    # Detectar caras en la imagen
    caras = dlib.get_frontal_face_detector()(imagen)

    # Si se detecta al menos una cara, extraer los encodings
    if caras:
        forma = predictor(imagen, caras[0])
        encoding = np.array(facial_recognition_model.compute_face_descriptor(imagen, forma))
        return encoding
    else:
        print("No se detectaron caras en la imagen.")
        return None

# Ruta de la imagen de ejemplo, sería una consulta a la base de datos de donde están registrados los alumnos
imagen_a_procesar = "referencia.jpeg"

# Extraer los encodings de la imagen, luego será una conversion de los datos ya existentes
encoding_imagen = extraer_encodings(imagen_a_procesar)

"""

 /$$                 /$$                                        
| $$                | $$                                        
| $$$$$$$   /$$$$$$ | $$  /$$$$$$        /$$$$$$/$$$$   /$$$$$$ 
| $$__  $$ /$$__  $$| $$ /$$__  $$      | $$_  $$_  $$ /$$__  $$
| $$  \ $$| $$$$$$$$| $$| $$  \ $$      | $$ \ $$ \ $$| $$$$$$$$
| $$  | $$| $$_____/| $$| $$  | $$      | $$ | $$ | $$| $$_____/
| $$  | $$|  $$$$$$$| $$| $$$$$$$/      | $$ | $$ | $$|  $$$$$$$
|__/  |__/ \_______/|__/| $$____/       |__/ |__/ |__/ \_______/
                        | $$                                    
                        | $$      
                        
                        
  _           _       
 | |         | |      
 | |__   ___ | | __ _ 
 | '_ \ / _ \| |/ _` |
 | | | | (_) | | (_| |
 |_| |_|\___/|_|\__,_|
                      
                                                 

"""







# Exrtaccion de descriptores en vivo  <<<<<<<<<<<<<<<<<<<



# Inicializar la cámara (puedes ajustar el índice de la cámara según tu configuración)
cap = cv2.VideoCapture(0)

# Variable para almacenar los descriptores faciales
descriptores_faciales = None


#mensaje_impreso = False

# Capturar descriptores faciales
while True:
    # Leer el frame desde la cámara
    ret, frame = cap.read()

    # Convertir a escala de grises para la detección facial
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el frame
    caras = detector(gray)

    # Si se detecta al menos un rostro
    if len(caras) > 0:
        # Obtener las coordenadas del primer rostro detectado
        x, y, w, h = caras[0].left(), caras[0].top(), caras[0].width(), caras[0].height()

        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Actualizar las coordenadas del rectángulo alrededor del rostro, ya con los valores de la pisicón
        coordenadas_recuadro = (x, y, w, h)

        # Dibujar el rectángulo alrededor del rostro
        if descriptores_faciales is not None:
         x, y, w, h = coordenadas_recuadro
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)



        # Extraer descriptores faciales solo si no se han capturado antes
        if descriptores_faciales is None:
            forma = predictor(gray, caras[0])
            descriptores_faciales = np.array(facial_recognition_model.compute_face_descriptor(frame, forma))
            print("Descriptores faciales capturados:")
           

        # Se calcula la distancia euclidiana entre los descriptores
        distancia = np.linalg.norm(encoding_imagen - descriptores_faciales)
   
         # Definir un umbral para decidir si las caras son lo suficientemente similares
         # 06 es un umbral medio va de 0.4 a 0.6 para no obteber falsos positivos 
         #Ya probé con 0.8 y corre estable 
       
        umbral = 0.6 
        
    
        match = "MATCH" if distancia < umbral else "Estudiante no registrado"
        print(match)
        
        
    
       

    # frame resultante
    cv2.imshow('Captura de Descriptores Faciales', frame)
    
    
    # tecla para cerrar el frame
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Con la r vuelve a calcular los encodings
        descriptores_faciales = None

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()