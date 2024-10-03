import matplotlib.pyplot as plt
import classes.Neuron as Neu
import utilities.neuronIdGiver as id
import classes.SoftMaxNeuron as sft
from tqdm import tqdm
import time
import utilities.getClasses as getClasses
from prettytable import PrettyTable
import numpy as np
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm


class Ambrogio:
    def __init__(self,usingALoadedModel=False) -> None:
        self.layers = []
        self.activations = []
        self.derivatives = []
        self.createStructure()
        self.newWeightsFromTraining = None
        self.velocity = [np.zeros_like(weights) for weights in self.getMatrixOfWeights()]
        self.momentum = 0.9
        if usingALoadedModel:
            self.loadState()
    
    def createStructure(self):
        with tqdm(total=100) as pbar:
            pbar.set_description("Creazione del layer di input...")
            # Create the neurons
            i = 200
            idGiver = id.IdGiver()
            
            # create the input layer with the same number of neurons as the number of features extracted from the images => 4096
            layer = [Neu.Neuron(idGiver.giveId()) for x in range(4096)]
            self.layers.append(layer)
            pbar.update(20)
            pbar.set_description("Creazione dei layer nascosti...")
            while i > 0:
                if i-2 == 0:
                    layer = [Neu.Neuron(idGiver.giveId()) for x in range(len(getClasses.getClasses()))]
                else:
                    layer = [Neu.Neuron(idGiver.giveId()) for x in range(i)]   
                
                self.layers.append(layer)
                i-=2
            pbar.update(50)
            pbar.set_description("Creazione del softmax neuron...")
            
            # create the softmax layer 
            self.layers.append([sft.SoftMaxNeuron()])

            pbar.update(10)
            pbar.set_description("Inizializzazione dei pesi...")
            
            for i,layer in enumerate(self.layers):
                if i == len(self.layers)-1:
                    break
                weigthMatrix = self.initialize_weights((len(layer),len(self.layers[i+1])))
                for j,neuron in enumerate(layer):
                    neuron.weights = weigthMatrix[j]
            pbar.update(10)
            pbar.set_description("ultimazione...")
            # save derivatives per layer
            self.inizializeDerivatives()
            pbar.update(5)
            # save activations per layer
            self.initializeActivations()
            pbar.update(5)
    
    def initializeActivations(self):
        activations = []
        for i in range(len(self.layers)):
            a = np.zeros(len(self.layers[i]))
            activations.append(a)
        self.activations = activations
    
    def inizializeDerivatives(self):
        derivatives = []
        for i in range(len(self.layers)-1):
            d = np.zeros((len(self.layers[i]), len(self.layers[i + 1])))
            derivatives.append(d)
        self.derivatives = derivatives

    def getNeurons(self):
        return self.layers
    
    def initialize_weights(self, shape):
        # Per sigmoid o tanh
        return np.random.randn(*shape) * np.sqrt(1. / shape[0])

        # Per ReLU
        # return np.random.randn(*shape) * np.sqrt(2. / shape[0])

    
    def getNeuron(self, id):
        for layer in self.layers:
            for neuron in layer:
                if neuron.getId() == id:
                    return neuron
        return None
    
    def predict(self, inputs): 
        self.initializeActivations() 
        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations
        # iterate through the network layers
        for i, w in enumerate(self.getMatrixOfWeights()):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)
            # apply sigmoid activation function
            activations = self.layers[0][0].sigmoid(net_inputs)
        
            # save the activations for backpropogation
            #if i+1 < len(self.activations):
            self.activations[i+1] = activations
        # return output layer activation
        
        predictions = self.layers[-1][0].calcFinalProbabilities(self.activations[-2])
        return predictions
        
                
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
    
    def getMatrixOfWeights(self):
        if self.newWeightsFromTraining is not None:
            return self.newWeightsFromTraining
        
        matrix = []
        for i,layer in enumerate(self.layers):
            if i == len(self.layers)-1:
                break
            layer = [neuron.weights for neuron in layer]
            layer = np.array(layer)
            matrix.append(layer)
        return matrix
    
    def back_propagate(self, error):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self.layers[0][0].sigmoid(activations,True)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]
            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)
            # backpropogate the next error
            error = np.dot(delta, self.getMatrixOfWeights()[i].T)


    def train(self, inputs, targets, epochs, learning_rate, momentum=False):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        manager = dsm.DataSetManager()
        
        errors = []
        # now enter the training loop
        for i in range(epochs):
            last_error = 0
            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                    # activate the network!
                output = self.predict(input)

                error = self.layers[-1][0].calcCrossEntropyLoss(target,output)

                self.back_propagate(error)

                    # now perform gradient descent on the derivatives
                    # this will update the weights
                if momentum:
                    self.gradient_descent_with_momentum(learning_rate)
                else:
                    self.gradient_descent(learning_rate)

                    # keep track of the error as we train
                last_error = error
            errors.append(last_error)
        plt.semilogy(range(len(errors)),errors)
        plt.xlabel('number of batches per epoch')
        plt.ylabel('Error')
        print("last error => ",last_error)
        plt.show()
            

        print("Training complete!")
        print("=====")



    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        newMatrix = self.getMatrixOfWeights()
        # update the weights by stepping down the gradient
        for i in range(len(self.getMatrixOfWeights())-1):
            weights = self.getMatrixOfWeights()[i]
            derivatives = self.derivatives[i]
            #print("before weights => ",weights)
            np.add(weights,derivatives * learningRate, out=weights, casting="unsafe")
            newMatrix[i] = weights
        self.newWeightsFromTraining = newMatrix

    
    """Aggiorna i pesi usando il gradient descent con momentum."""
    def gradient_descent_with_momentum(self, learningRate=1):
        
        newMatrix = self.getMatrixOfWeights()
        # aggiorna i pesi scendendo lungo il gradiente con momentum
        for i in range(len(self.getMatrixOfWeights()) - 1):
            weights = self.getMatrixOfWeights()[i]
            derivatives = self.derivatives[i]
            
            # Aggiorna la velocità con momentum
            self.velocity[i] = self.momentum * self.velocity[i] - learningRate * derivatives
            
            # Aggiorna i pesi usando la velocità
            np.add(weights, self.velocity[i], out=weights, casting="unsafe")
            
            newMatrix[i] = weights
        
        self.newWeightsFromTraining = newMatrix
            
            
    def loadState(self):
            data = np.load('state.npz')
            matrix = []
            for i in range(len(self.layers)-1):
                matrix.append(data[f'matrix{i+1}'])
            for i,layer in enumerate(self.layers):
                if i == len(self.layers)-1:
                    break
                for j,neuron in enumerate(layer):
                    neuron.weights = matrix[i][j]
        
    
    def saveState(self):
        m = self.getMatrixOfWeights()
        np.savez('state.npz', matrix1=m[0], matrix2=m[1], matrix3=m[2], matrix4=m[3], matrix5=m[4], matrix6=m[5], matrix7=m[6], matrix8=m[7], matrix9=m[8], matrix10=m[9], matrix11=m[10])