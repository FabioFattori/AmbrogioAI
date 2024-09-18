import classes.Connection as c
import random
import classes.Neuron as Neu
import utilities.neuronIdGiver as id
import classes.SoftMaxNeuron as sft
import classes.Cacher as Cacher
from tqdm import tqdm
import time
import utilities.getClasses as getClasses
from prettytable import PrettyTable
import numpy as np
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm

class Ambrogio:
    def __init__(self):
        self.neurons = []
        self.cacher = Cacher.Cacher()
        self.createStructure()
    
    def createStructure(self):
        # Create the neurons
        i = 20
        idGiver = id.IdGiver()
        while i > 0:
            if i-2 == 0:
                layer = [Neu.Neuron(random.randrange(0,10),idGiver.giveId()) for x in range(len(getClasses.getClasses()))]
            else:
                layer = [Neu.Neuron(random.randrange(0,10),idGiver.giveId()) for x in range(i)]   
                
            self.neurons.append(layer)
            if len(self.neurons) > 1:
                for (neuron) in self.neurons[-2]:
                    for n in layer:
                        randomValue = random.randrange(0,10)
                        connection = c.Connection(n, randomValue, neuron)
                        n.appendGiver(connection)
                        connection = c.Connection(neuron, randomValue, n)
                        neuron.appendTaker(connection)
            i-=2
        self.createSoftmaxLayer()
            
    def createSoftmaxLayer(self):
        softmax = sft.SoftMaxNeuron()
        for neuron in self.neurons[-1]:
            randomValue = 1
            connection = c.Connection(softmax, randomValue, neuron)
            neuron.appendTaker(connection)
            softmax.appendGiver(connection)
        self.neurons.append([softmax])
        
    def getNeurons(self):
        return self.neurons
    
    def getNeuron(self, id):
        for layer in self.neurons:
            for neuron in layer:
                if neuron.getId() == id:
                    return neuron
        return None
    
    def getNeuronLayer(self, id):
        for layer in self.neurons:
            for neuron in layer:
                if neuron.getId() == id:
                    return layer
        return None
    
    def predict(self, inputs):
        self.cacher = Cacher.Cacher()
        for i in range(min(len(inputs),len(self.neurons[0]))):
            self.neurons[0][i].receiveData(inputs[i])
        with tqdm(total=100) as pbar:
            for layer in self.neurons[1:len(self.neurons)-1]:
                for neuron in layer:
                    neuron.output(self.cacher)
                pbar.update(100/len(self.neurons))
            pbar.update(100-pbar.n)
        self.showPrediction(self.neurons[-1][0].output(self.cacher))
        return self.neurons[-1][0].output(self.cacher)
        
                
    def __str__(self) -> str:
        toRet = "Ambrogio AI"
        for layer in self.neurons:
            toRet += "\n"
            for neuron in layer:
                toRet += f"\n{neuron}"
        return toRet
    
    def showPrediction(self,predictions):
        table = PrettyTable()
        table.title = "Probabilità di appartenenza alle classi"
        table.align = "l"
        table.border = True
        table.header = True
        
        # Aggiungi le colonne
        table.field_names = ["Etichetta", "Probabilità"]
        for i in range(len(predictions)):
            table.add_row([getClasses.getClasses()[i],predictions[i]])

        # Stampa la tabella
        print(table)
        
        print(f"La classe predetta è: {getClasses.getClasses()[np.argmax(predictions)]}")
    
    def train(self,filePaths,learningRate = 0.01,epocs = 2):
        with tqdm(total=100) as pbar:    
            for epoc in range(epocs):
                print(f"Epoca {epoc+1}/{epocs}")
                for path in filePaths:
                    featureMap = fe.FeatureExtractor().extract_features(path)
                    self.predict(featureMap)
                    self.backpropagation(learningRate,path)
                pbar.update(1)
    
    def backpropagation(self, learningRate, path):
        # Calcolo dell'errore
        y_true = dsm.DataSetManager().getCorrentPredictionOfImage(path)
        y_pred = self.neurons[-1][0].output(self.cacher)
        error = self.neurons[-1][0].calcCrossEntropyLoss(y_true, y_pred)
        print("Errore: ", error)

        print("y => ", y_true)
        print("y_hat => ", y_pred)

        # check if the error is nan or inf
        if np.isnan(error) or np.isinf(error):
            print("Errore nan o inf")
            return

        # Calcolo del gradiente dell'errore rispetto all'output
        delta = y_pred - y_true

        # Aggiornamento dei pesi
        for i in reversed(range(len(self.neurons))):
            layer = self.neurons[i]
            new_delta = np.zeros(len(layer))  # Delta per il layer precedente
            for j, neuron in enumerate(layer):
                neuron_output = neuron.output(self.cacher)  # Output del neurone attuale
                for giver in neuron.getGivers():
                    # Calcolo del gradiente rispetto al peso
                    gradient = delta[j] * giver.from_.output(self.cacher)
                    # Aggiornamento del peso
                    giver.weight -= learningRate * gradient
                    # Calcolo del nuovo delta per il layer precedente (derivata sigmoid o altra attivazione)
                    new_delta[j] += delta[j] * giver.weight * neuron_output * (1 - neuron_output)
            delta = new_delta  # Passa al layer precedente

        # Reset del cacher
        self.cacher = Cacher.Cacher()
    
        
    
    def getConnectionsOfNeuron(self,neuron):
        return neuron.getGivers()