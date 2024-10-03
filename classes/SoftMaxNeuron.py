import numpy as np
import classes.Neuron as Neu
import utilities.getClasses as getClasses

class SoftMaxNeuron(Neu.Neuron):
    
    def __init__(self) -> None:
        self.id = "SoftMaxNeuron"
        self.inputs = []
        self.weights = [0 for x in range(len(getClasses.getClasses()))]
        self._output = None
        pass

    def receiveData(self, data):
        self.inputs = data
    

    def calcCrossEntropyLoss(self, y, y_hat):
        # print("y => ",y)
        # return -np.log(np.sum(y * y_hat))
        # computing softmax values for predicted values
        loss = 0
        
        for i in range(len(y_hat)):
            loss = loss + (-1 * y[i]*np.log(y_hat[i]))
    
        return loss

    
    def calcFinalProbabilities(self, y):
        #print("output of NN => ",y)
        e_x = np.exp(y - np.max(y))
        return e_x / e_x.sum(axis=0)
    
    def output(self):
        if self._output is None:
            self._output = self.calcFinalProbabilities(self.inputs)
        return self._output
    
    def __str__(self) -> str:
        return "SoftMaxNeuron"