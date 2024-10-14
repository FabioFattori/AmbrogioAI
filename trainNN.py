import classes.AmbrogioSimple as A
from utilities import getClasses
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm
from tqdm import tqdm

dataSet = dsm.DataSetManager()
featureExtractor = fe.FeatureExtractor()
ambrogio = A.AmbrogioSimple()

print("si vuole caricare lo stato della rete neurale salvato se presente? [y/n]")
if input() == 'y':
    
    with tqdm(total=100) as pbar:
        ambrogio.loadState()
        pbar.update(100)

inputs = dataSet.randomShuffleDataSet()
targets = [dataSet.getCorrentPredictionOfImage(image) for image in inputs]
inputs = [featureExtractor.extract_features(path) for path in inputs]

ambrogio.train(inputs,targets)

print("si vuole salvare lo stato della rete neurale? [y/n]")
try:
    if input() == 'y':
        with tqdm(total=100) as pbar:
            ambrogio.saveState()
            pbar.update(100)
except Exception as e:
    print("non ho capito")



