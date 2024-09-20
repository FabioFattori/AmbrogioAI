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
        
        # create the input layer with the same number of neurons as the number of features extracted from the images => 4096
        layer = [Neu.Neuron(random.randrange(0,10),idGiver.giveId()) for x in range(4096)]
        self.neurons.append(layer)
        
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
    