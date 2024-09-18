import utilities.getClasses as getClasses
import random
import os

class DataSetManager:
    def __init__(self):
        self.imagesPath = "imgs/"
        self.jsonPath = "imgs\dataSet.json"
        
    def getAllImages(self):
        # get all the images from the imgs folder
        classes = getClasses.getClasses()
        images = []
        for c in classes:
            pathChosen = self.imagesPath + c + "/"
            images += [pathChosen + i for i in os.listdir(pathChosen)]
        
        return images
    
    def getCorrentPredictionOfImage(self,imagePath):
        # get the correct prediction of the image
        classes = getClasses.getClasses()
        for c in classes:
            if c in imagePath:
                index = imagePath.index(c)
                toRet = [0 for x in range(len(classes))]
                toRet[classes.index(c)] = 1
                return toRet
        return None
    
    # NOT TESTED FUNCTION
    def getImageRandomImage(self):
        # go in the imgs folder and get a random image from a random class
        classes = getClasses.getClasses()
        randomClass = random.choice(classes)
        pathChosen = self.imagesPath + randomClass + "/"
        images = os.listdir(pathChosen)
        
        return pathChosen + random.choice(images)        
        