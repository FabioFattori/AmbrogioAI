import classes.Ambrogio as A
from utilities import getClasses
import utilities.drawANN as draw
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm
from tqdm import tqdm
import classes.SoftMaxNeuron as sft
import random

print(getClasses.getClasses())

ambrogio = A.Ambrogio()


#ambrogio.predict(fe.FeatureExtractor().extract_features(dsm.DataSetManager().getRandomImage()))
#[print("layer {} neuron : {}".format(i,neuron)) for i,layer in enumerate(ambrogio.getNeurons()) for neuron in layer]

dataSet = dsm.DataSetManager()
featureExtractor = fe.FeatureExtractor()

inputs = [dataSet.getAllImages()[0]]
path = inputs
targets = [dataSet.getCorrentPredictionOfImage(image) for image in inputs]
inputs = [featureExtractor.extract_features(path) for path in inputs]

ambrogio.train(inputs,targets,20,1)
print(path[0])
pred = ambrogio.predict(inputs)
print(pred)
print(ambrogio.activations[10])


ambrogio.showPrediction(ambrogio.getNeuron("SoftMaxNeuron").calcFinalProbabilities(pred))

print("si vuole disegnare la rete neurale? [y/n]")
if input() == 'y':
    try:
        with tqdm(total=100) as pbar:
            layers = ambrogio.getNeurons()
            g = draw.draw_neural_network(layers)
            pbar.update(50)
            g.render('Ambrogio', view=True,format='png')
            pbar.update(50)
        
    except Exception as e:
        print(e)
        print("Errore nel disegno del grafo del modello, probabilmente non hai inserito nel path il programma dot di graphviz")
else:
    print("ok, non disegno il grafo")