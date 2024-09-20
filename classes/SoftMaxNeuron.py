import numpy as np

class SoftMaxNeuron:
    
    def __init__(self) -> None:
        self.id = "SoftMaxNeuron"
        self.dataGivers = []
        pass
    
    def calcCrossEntropyLoss(self, y, y_hat):
        return -np.log(np.sum(y * y_hat))

    
    def calcFinalProbabilities(self, y):
        e_x = np.exp(y - np.max(y))
        return e_x / e_x.sum(axis=0)
    
    def output(self,cacher):
        out = np.empty(len(self.dataGivers))
        for giver in self.dataGivers:
            np.append(out,cacher.get(giver.from_.id))
            
        return self.calcFinalProbabilities(out)
    
    def appendGiver(self,giver):
        if giver not in self.dataGivers:
            self.dataGivers.append(giver)
    
    def getGivers(self):
        return self.dataGivers
    
    def __str__(self) -> str:
        return "SoftMaxNeuron"