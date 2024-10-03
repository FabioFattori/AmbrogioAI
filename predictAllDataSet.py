import classes.Ambrogio as A
from utilities import getClasses
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm
import numpy as np


print(getClasses.getClasses())
ambrogio = A.Ambrogio()

print("si vuole caricare lo stato della rete neurale salvato se presente? [y/n]")
if input() == 'y':
    ambrogio.loadState()
    

dataSet = dsm.DataSetManager()
featureExtractor = fe.FeatureExtractor()

inputs = dataSet.randomShuffleDataSet()
print(len(inputs))
targets = [dataSet.getCorrentPredictionOfImage(image) for image in inputs]
inputs = [featureExtractor.extract_features(path) for path in inputs]

for i in range(len(inputs)):
    out = ambrogio.predict(inputs[i])
    #ambrogio.showPrediction(out)
    error = ambrogio.layers[-1][0].calcCrossEntropyLoss(targets[i],out)
    print("ERROR => ",error)
    print("output => ",out)
    print("desired output => ",targets[i])
    print("--------------------------------------------------")

