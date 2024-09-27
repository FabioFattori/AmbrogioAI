import classes.Ambrogio as A
from utilities import getClasses
import utilities.drawANN as draw
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm
from tqdm import tqdm
import classes.SoftMaxNeuron as sft
import numpy as np
import random

print(getClasses.getClasses())

ambrogio = A.Ambrogio()

print("si vuole caricare lo stato della rete neurale salvato se presente? [y/n]")
if input() == 'y':
    
    with tqdm(total=100) as pbar:
        ambrogio.loadState()
        pbar.update(100)


#ambrogio.predict(fe.FeatureExtractor().extract_features(dsm.DataSetManager().getRandomImage()))
#[print("layer {} neuron : {}".format(i,neuron)) for i,layer in enumerate(ambrogio.getNeurons()) for neuron in layer]

dataSet = dsm.DataSetManager()
featureExtractor = fe.FeatureExtractor()

inputs = dataSet.getAllImages()

targets = [dataSet.getCorrentPredictionOfImage(image) for image in inputs]
inputs = [featureExtractor.extract_features(path) for path in inputs]

# for input in enumerate(inputs):
#     print(input[1])
#     #print(np.max(input))
#     print("====")
# exit()

# print("input : ",path)
# pred = ambrogio.predict(inputs[0])

# ambrogio.showPrediction(pred)

ambrogio.train(inputs,targets,1000,0.01)

inputs = dataSet.getRandomImage()
print(inputs)
out = ambrogio.predict(featureExtractor.extract_features(inputs))
ambrogio.showPrediction(out)
print(dataSet.getCorrentPredictionOfImage(inputs))
print(getClasses.getClasses())

print("si vuole salvare lo stato della rete neurale? [y/n]")
try:
    if input() == 'y':
        with tqdm(total=100) as pbar:
            ambrogio.saveState()
            pbar.update(100)
except Exception as e:
    print("non ho capito")
    

    

# print("si vuole disegnare la rete neurale? [y/n]")
# if input() == 'y':
#     try:
#         with tqdm(total=100) as pbar:
#             layers = ambrogio.getNeurons()
#             g = draw.draw_neural_network(layers)
#             pbar.update(50)
#             g.render('Ambrogio', view=True,format='png')
#             pbar.update(50)
        
#     except Exception as e:
#         print(e)
#         print("Errore nel disegno del grafo del modello, probabilmente non hai inserito nel path il programma dot di graphviz")
# else:
#     print("ok, non disegno il grafo")


