import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

# Rediriger stdout et stderr vers un fichier
sys.stdout = open('output.log', 'w', encoding='utf-8')
sys.stderr = open('error.log', 'w', encoding='utf-8')
# Charger le modèle pré-entraîné
model = load_model('plastic_detection_model.h5')

# Fonction pour prédire et dessiner les résultats
def detect_plastic(frame, model, threshold=0.9):
    height, width, _ = frame.shape
    step_size = 50  # Taille de l'étape pour découper l'image
    box_size = 150  # Taille de la boîte pour la prédiction

    # Parcourir l'image en utilisant un pas pour diviser l'image
    for y in range(0, height - box_size, step_size):
        for x in range(0, width - box_size, step_size):
            # Découper la sous-image
            sub_image = frame[y:y + box_size, x:x + box_size]
            sub_image_resized = cv2.resize(sub_image, (150, 150)) / 255.0
            sub_image_expanded = np.expand_dims(sub_image_resized, axis=0)

            # Prédire
            prediction = model.predict(sub_image_expanded)
            if prediction[0] > threshold:  # Si le seuil est dépassé
                # Dessiner un rectangle vert autour de la zone détectée
                cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (0, 255, 0), 2)

    return frame

# Capture vidéo en temps réel
cap = cv2.VideoCapture(0)  # 0 pour la caméra par défaut

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détection de plastique
    result_frame = detect_plastic(frame, model)

    # Afficher le cadre avec détection
    cv2.imshow('Plastic Detection', result_frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
