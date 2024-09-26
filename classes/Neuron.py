import numpy as np

class Neuron:
    def __init__(self,id):
        self._output = None
        self.weights = []
        self.id = id
        self.input = 0

    def getId(self):
        return self.id
        
    def receiveData(self,data):
        self.input = data

    def sigmoid(self,x,deriv=False):
        if deriv:
            return x * (1.0 - x)
        #print("x => ",x)
        toret = 1.0 / (1.0 + np.exp(-x))
        return toret
    
    def ReLu(self,x,derivative=False):
        if derivative:
            return 1 * (x > 0)  #returns 1 for any x > 0, and 0 otherwise
        
        return max(0, x)
            
    def output(self):
        #check if the input is a list or a single value
        if type(self.input) != list:
            self._output = self.input
        elif self._output is not None:
            return self._output
        else:
            self._output = 0
            for i in range(len(self.weights)-1):
                self._output += self.weights[i] * self.input[i]
            
        self._output = self.sigmoid(self._output)
        return self._output
    
    def __str__(self) -> str:
        toRet = f"Neuron {self.id}"

        return toRet