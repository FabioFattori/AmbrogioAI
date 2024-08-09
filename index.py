import classes.Ambrogio as A
from utilities import getClasses
import utilities.drawANN as draw
import classes.FeatureExtractor as fe

print(getClasses.getClasses())

ambrogio = A.Ambrogio()

pathToFile = "imgs\dark-souls-2-scholar-of-the-firs.png"

featureMap = fe.FeatureExtractor().extract_features(pathToFile)

print("features estratte => ",featureMap)

prediction = ambrogio.predict(featureMap)


layers = ambrogio.neurons
g = draw.draw_neural_network(layers)
#g.render('Ambrogio', view=True)


#print(ambrogio)