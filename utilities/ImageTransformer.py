from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, target_func=None):
        """
        image_paths: Lista dei path delle immagini.
        transform: Trasformazioni da applicare sulle immagini.
        target_func: Funzione per ottenere i target (es: getCorrentPredictionOfImage).
        """
        self.image_paths = image_paths
        self.transform = transform
        self.target_func = target_func

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Carica l'immagine e convertila in RGB

        # Ottieni l'etichetta usando la funzione di target (getCorrentPredictionOfImage)
        label = self.target_func(img_path)
        label = label.index(1)  # Converti one-hot in indice di classe

        if self.transform:
            image = self.transform(image)

        return image, label
