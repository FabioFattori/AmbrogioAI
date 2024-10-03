import classes.Ambrogio as A
from utilities import getClasses
import utilities.drawANN as draw
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm
from tqdm import tqdm
import classes.SoftMaxNeuron as sft

print(getClasses.getClasses())

dataSet = dsm.DataSetManager()
featureExtractor = fe.FeatureExtractor()
ambrogio = A.Ambrogio()

print("si vuole caricare lo stato della rete neurale salvato se presente? [y/n]")
if input() == 'y':
    
    with tqdm(total=100) as pbar:
        ambrogio.loadState()
        pbar.update(100)

inputs = dataSet.randomShuffleDataSet()
targets = [dataSet.getCorrentPredictionOfImage(image) for image in inputs]
inputs = [featureExtractor.extract_features(path) for path in inputs]

ambrogio.train(inputs,targets,10,1,True)
ambrogio.train(inputs,targets,100,0.1,True)
ambrogio.train(inputs,targets,200,0.01,True)

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


