#CAPTURA DE DESCRIPTORES SOLO IMAGEN


import dlib
import numpy as np

# Cargar el modelo de predicción facial de dlib
predictor = dlib.shape_predictor("env/Lib/site-packages/dlib/models/shape_predictor_68_face_landmarks.dat")  # Asegúrate de descargar este modelo
facial_recognition_model = dlib.face_recognition_model_v1("env/Lib/site-packages/dlib/models/dlib_face_recognition_resnet_model_v1.dat")  # Asegúrate de descargar este modelo

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
encoding_imagen = extraer_encodings(imagen_a_procesar)

# Imprimir el encoding (puedes almacenarlo o compararlo con otros encodings según tus necesidades)
if encoding_imagen is not None:
    print("Encoding de la imagen:", encoding_imagen)
