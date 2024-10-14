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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
    def optimizerResolver(self,mode:Optimazer) -> optim:
        if mode == Optimazer.Adam:
            return optim.Adam
        elif mode == Optimazer.RMSprop:
            return optim.RMSprop
        elif mode == Optimazer.StochasticGradientDescent:
            return optim.SGD
        
        raise Exception("Optimizer not supported")
    

    # TODO implementare il metodo di training del modello