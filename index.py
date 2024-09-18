import classes.Ambrogio as A
from utilities import getClasses
import utilities.drawANN as draw
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm

print(getClasses.getClasses())

ambrogio = A.Ambrogio()

pathToFile = "imgs/sportivo/s1.png"

featureMap = fe.FeatureExtractor().extract_features(pathToFile)

print("features estratte => ",featureMap)

prediction = ambrogio.predict(featureMap)


layers = ambrogio.neurons

try:
    g = draw.draw_neural_network(layers)

    g.render('Ambrogio', view=True)
except Exception as e:
    print(e)
    print("Errore nel disegno del grafo del modello, probabilmente non hai inserito nel path il programma dot di graphviz")


#print(ambrogio)

filePaths = dsm.DataSetManager().getAllImages()

print(filePaths)

ambrogio.train(filePaths,0.05)

ambrogio.predict(featureMap)
