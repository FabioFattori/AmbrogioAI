import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import utilities.getClasses as gc
import utilities.DataSetManager as dsm
from enum import Enum


class Optimazer(Enum):
    """ottimizzatori supportati dal modello"""
    Adam = 0
    RMSprop = 1
    StochasticGradientDescent = 2
    

class AmbrogioNet50:
    def __init__(self,lr=0.001, momentum=0.9, optimizer=Optimazer.StochasticGradientDescent, step_size=7, gamma=0.1):
        
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Sostituire l'ultimo fully connected layer (fc)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(gc.getClasses()))  # n classi in output layer 
        
        self.setDevice()
        # imposta la loss function
        self.criterion = nn.CrossEntropyLoss()
        # imposta l'ottimizzatore
        self.optimizer = self.optimizerResolver(optimizer)(self.model.parameters(), lr=lr, momentum=momentum)
        # imposta il learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
                
    def setDevice(self):
        # Verifica se è disponibile la GPU e imposta il device di conseguenza
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def optimizerResolver(self,mode:Optimazer) -> optim:
        if mode == Optimazer.Adam:
            return optim.Adam
        elif mode == Optimazer.RMSprop:
            return optim.RMSprop
        elif mode == Optimazer.StochasticGradientDescent:
            return optim.SGD
        
        raise Exception("Optimizer not supported")
    
    import utilities.DataSetManager as dsm

    def train_model(self, num_epochs=25):
        dataManager = dsm.DataSetManager()
        dataloaders, dataset_sizes = dataManager.getSetForRes50()
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Ogni epoca ha una fase di addestramento e una di validazione
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Imposta il modello in modalità addestramento
                else:
                    self.model.eval()   # Imposta il modello in modalità valutazione

                running_loss = 0.0
                running_corrects = 0

                # Itera sui dati
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Azzera i gradienti dei parametri
                    self.optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # Backward + ottimizzazione solo nella fase di addestramento
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistiche
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        #return self.model
    
    def save_model(self, path = "AmbrogioResNet50.pth"):
        torch.save(self.model.state_dict(), path)