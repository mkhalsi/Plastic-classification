import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys

# Rediriger stdout et stderr vers un fichier
sys.stdout = open('output.log', 'w', encoding='utf-8')
sys.stderr = open('error.log', 'w', encoding='utf-8')

# Chemins des dossiers
plastique_path = 'C:/Users/SIGMA ITD/Downloads/Garbage classification/plastique'
non_plastique_path = 'C:/Users/SIGMA ITD/Downloads/Garbage classification/non_plastique'

# Fonction pour charger et prétraiter les images
def load_and_preprocess_images(folder_path, label, image_size=(150, 150)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        # Vérifier si l'image a été chargée correctement
        if img is not None:
            # Redimensionner et normaliser l'image
            img_resized = cv2.resize(img, image_size) / 255.0
            images.append(img_resized)
            labels.append(label)
            
    return np.array(images), np.array(labels)

# Chargement et préparation des images
plastique_images, plastique_labels = load_and_preprocess_images(plastique_path, 1)
non_plastique_images, non_plastique_labels = load_and_preprocess_images(non_plastique_path, 0)

# Combiner toutes les images et labels
all_images = np.concatenate((plastique_images, non_plastique_images), axis=0)
all_labels = np.concatenate((plastique_labels, non_plastique_labels), axis=0)



# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Construire le modèle CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Classification binaire

# Compilation du modèle
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Sauvegarde du modèle
model.save('plastic_detection_model.h5')

# Tracer les courbes de précision et de perte
plt.figure(figsize=(12, 4))

# Précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Afficher les graphiques
plt.show()

