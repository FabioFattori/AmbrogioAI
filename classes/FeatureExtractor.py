import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from keras.preprocessing import image # type: ignore
from keras.models import Model # type: ignore

class FeatureExtractor:

    def __init__(self) -> None:
        # Carica il modello VGG16 pre-addestrato e rimuove l'ultimo strato
        self.model = VGG16(weights='imagenet', include_top=True)
        self.model = Model(inputs=self.model.input, outputs=self.model.get_layer('fc2').output)
        
    def extract_features(self, img_path):
        """
        Estrae le caratteristiche da un insieme di immagini.

        :param img_paths: Lista dei percorsi delle immagini.
        :return: Caratteristiche ridotte a 2D => narray di lunghezza 4096.
        """
        features_list = []
        
        # Verifica se il file esiste
        if not os.path.isfile(img_path):
            print(f"File not found: {img_path}")
            
            
        # Carica l'immagine e la prepara per il modello
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        # Estrai le caratteristiche
        features = self.model.predict(img_data)
        features_list.append(features.flatten())  # Appiattisci le caratteristiche
        
        # Converti in array NumPy
        features_array = np.array(features_list)
        features_array = self.normalize(features_array)
        # Riduci la dimensione delle feature estratte a 2D e visualizzale
        return np.array(features_array.flatten())
    
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    
