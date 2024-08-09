
class Neuron:
    def __init__(self,bias,id):
        self.bias = bias
        self.dataGivers = []
        self.dataTakers = []
        self.id = id
        self.input = 0
        
    def receiveData(self,data):
        self.input = data
        
    def passDataToTakers(self):
        for taker in self.dataTakers:
            out = self.output()
            taker.receiveData(out)
            
    def output(self,cacher):
        if cacher.keyExists(self.id):
            return cacher.get(self.id)
        else:
            out = self.input + self.bias
            for giver in self.dataGivers:
                out += giver.from_.output(cacher) * giver.weight
                cacher.set(giver.from_.id, max(0, out + self.bias))
            
        #print(f"Neuron {self.id} output: {max(0, out + self.bias)}")
        
        return max(0, out + self.bias)
    
    def getGivers(self):
        return self.dataGivers
    
    def getTakers(self):
        return self.dataTakers
    
    def setGivers(self,givers):
        self.dataGivers = givers
    
    def setTakers(self,takers):
        self.dataTakers = takers
        
    def appendGiver(self,giver):
        if giver not in self.dataGivers and giver not in self.dataTakers:
            self.dataGivers.append(giver)
    
    def appendTaker(self,taker):
        if taker not in self.dataTakers and taker not in self.dataGivers:
            self.dataTakers.append(taker)
    
    
    def __str__(self) -> str:
        toRet = f"Neuron {self.id} with bias: {self.bias}"
        toRet += "\nData Givers:"
        for giver in self.dataGivers:
            toRet += f"\n\t{giver}"
        toRet += "\nData Takers:"
        for taker in self.dataTakers:
            toRet += f"\n\t{taker}"
        return toRet