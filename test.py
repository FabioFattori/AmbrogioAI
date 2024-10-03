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

inputs = dataSet.getRandomImage()
target = dataSet.getCorrentPredictionOfImage(inputs)
print(inputs)
out = ambrogio.predict(featureExtractor.extract_features(inputs))
#ambrogio.showPrediction(out)
error = ambrogio.layers[-1][0].calcCrossEntropyLoss(target,out)
print("ERROR => ",error)
print("output => ",out)
print("desired output => ",target)
