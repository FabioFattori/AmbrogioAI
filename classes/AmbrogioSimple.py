import utilities.DataSetManager as dsm
import classes.FeatureExtractor as fe
import numpy as np
import utilities.getClasses as getClasses
from prettytable import PrettyTable

input_size = 4096

class AmbrogioSimple():
    '''
    AmbrogioSimple è una rete neurale con un solo strato nascosto, 4096 neuroni di input e n neuroni nello strato nascosto (n=64 di default).
    '''
    def __init__(self, hidden_size=64, learning_rate=0.01):
        # Inizializzazione dei pesi
        self.learning_rate = learning_rate
        
        # Pesi per lo strato input -> hidden
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        
        output_size = len(getClasses.getClasses())
        
        # Pesi per lo strato hidden -> output
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
    
    # Funzione di attivazione ReLU
    def relu(self, x):
        return np.maximum(0, x)
    
    # Derivata della funzione ReLU
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    # Funzione softmax
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Funzione di perdita cross-entropy
    def cross_entropy_loss(self, predictions, targets):
        n_samples = targets.shape[0]
        clipped_predictions = np.clip(predictions, 1e-12, 1 - 1e-12)  # Per evitare problemi di log(0)
        log_likelihood = -np.log(clipped_predictions[range(n_samples), targets])
        loss = np.sum(log_likelihood) / n_samples
        return loss
    
    # Funzione per eseguire la forward propagation
    def forward(self, X):
        # Input -> Hidden
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        
        # Hidden -> Output
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.softmax(self.output_input)
        
        return self.output
    
    # Backward propagation e aggiornamento dei pesi
    def backward(self, X, y):
        n_samples = X.shape[0]
        
        # Calcolo del gradiente per l'output
        output_error = self.output
        output_error[range(n_samples), y] -= 1
        output_error /= n_samples
        
        # Gradienti per pesi e bias hidden -> output
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)
        
        # Gradiente per lo strato hidden
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_input)
        
        # Gradienti per pesi e bias input -> hidden
        d_weights_input_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)
        
        # Aggiornamento dei pesi con discesa del gradiente
        self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden
        self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
        self.bias_output -= self.learning_rate * d_bias_output
    
    # Funzione di allenamento
    def train(self, X, y, epochs=1000):
        # setup iniziale dei dati per il training
        if X is not np.array:
            X = np.array(X)
        
        if type(y[0]) is list:
            man = dsm.DataSetManager()
            y = man.convertPassedTargetsToTrainingTargets(y)
        
        if y is not np.array:
            y = np.array(y)
        
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Calcolo della perdita
            loss = self.cross_entropy_loss(predictions, y)
            
            # Backward pass e aggiornamento pesi
            self.backward(X, y)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
    
    # Predizione
    def predict(self, X):
        predictions = self.forward(X)
        return predictions[0]
    
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
    
    def saveState(self):
        np.savez('AmbrogioSimple', weights_input_hidden=self.weights_input_hidden, bias_hidden=self.bias_hidden,weights_hidden_output=self.weights_hidden_output, bias_output=self.bias_output)
    
    def loadState(self):
        data = np.load('AmbrogioSimple.npz')
        self.weights_input_hidden = data['weights_input_hidden']
        self.bias_hidden = data['bias_hidden']
        self.weights_hidden_output = data['weights_hidden_output']
        self.bias_output = data['bias_output']
    
    def getLayers(self):
        layers = []
        layers.append(self.weights_input_hidden)
        layers.append(self.weights_hidden_output)
        return layers

if __name__ == '__main__':
    # y_train = np.random.randint(0, 3, size=100)
    # print(y_train)
    # exit()
    # Esempio di utilizzo
    man = dsm.DataSetManager()
    fea = fe.FeatureExtractor()
    
    X = man.randomShuffleDataSet()
    test = fea.extract_features(X.pop(0))
    y = [man.getCorrentPredictionOfImage(image) for image in X]
    
    for i in range(len(y)):
        y[i] = y[i].index(1)
    print(y)
    
    X = [fea.extract_features(image) for image in X]
    X = np.array(X)
    y = np.array(y)
    
    nn = AmbrogioSimple()
    nn.train(X, y)
    
    print('Predizione giusta => {}'.format(man.getCorrentPredictionOfImage(man.getAllImages()[0])))
    ouyt = nn.predict(test)
    print(ouyt)
    
    
    for path in man.getAllImages():
        nn.showPrediction(nn.predict(fea.extract_features(path)))
        print('Predizione giusta => {}'.format(getClasses.getClasses()[np.argmax(man.getCorrentPredictionOfImage(path))]))
        print()