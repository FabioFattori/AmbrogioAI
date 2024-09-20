import classes.Ambrogio as A
from utilities import getClasses
import utilities.drawANN as draw
import classes.FeatureExtractor as fe
import utilities.DataSetManager as dsm
from tqdm import tqdm


print(getClasses.getClasses())

ambrogio = A.Ambrogio()

pathToFile = "imgs/sportivo/s1.png"


featureMap = fe.FeatureExtractor().extract_features(dsm.DataSetManager().getAllImages()[0])


prediction = ambrogio.predict(featureMap)

layers = ambrogio.neurons

print("si vuole disegnare la rete neurale? [y/n]")
if input() == 'y':
    try:
        with tqdm(total=100) as pbar:
            g = draw.draw_neural_network(layers)
            pbar.update(50)
            g.render('Ambrogio', view=True,format='png')
            pbar.update(50)
        
    except Exception as e:
        print(e)
        print("Errore nel disegno del grafo del modello, probabilmente non hai inserito nel path il programma dot di graphviz")
else:
    print("ok, non disegno il grafo")