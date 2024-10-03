import utilities.getClasses as getClasses
import random
import os
import numpy as np
import classes.FeatureExtractor as fe

class DataSetManager:
    def __init__(self):
        self.imagesPath = "imgs/"
        self.jsonPath = "imgs\dataSet.json"
    
    def getAllImages(self):
        '''
        get the entire dataset in a form of a list of image paths in ORDER 
        '''
        # get all the images from the imgs folder
        classes = getClasses.getClasses()
        images = []
        for c in classes:
            pathChosen = self.imagesPath + c + "/"
            images += [pathChosen + i for i in os.listdir(pathChosen)]
        
        for i,path in enumerate(images):
            # check if in path is contained .DS_Store
            if ".DS_Store" in path:
                images.pop(i)
        
        return images
    
    def getCorrentPredictionOfImage(self,imagePath):
        '''
        given the path of the image, return the correct prediction of the image that ambrogio should return
        '''
        # get the correct prediction of the image
        classes = getClasses.getClasses()
        for c in classes:
            if c in imagePath:
                toRet = [0 for x in range(len(classes))]
                toRet[classes.index(c)] = 1
                return toRet
        return None
    
    def getRandomImage(self):
        '''
        get a random image path of the dataset
        '''
        # go in the imgs folder and get a random image from a random class
        classes = getClasses.getClasses()
        randomClass = random.choice(classes)
        pathChosen = self.imagesPath + randomClass + "/"
        images = os.listdir(pathChosen)
        
        return pathChosen + random.choice(images)        
    
    def partitionDataSet(self):
        '''
        The data will be a tuple
        return 0 => the training set, 1 => the convalidation 
        '''
        # partition the data set into training and test set
        images = self.getAllImages()
        random.shuffle(images)
        trainingSet = images[:int(len(images)*0.7)]
        convalidationSet = images[int(len(images)*0.7):]
        
        
        return trainingSet,convalidationSet
    
    def randomShuffleDataSet(self):
        '''
        get all the data set like 'getAllImages' but shuffled randomly
        '''
        # shuffle the dataset
        images = self.getAllImages()
        random.shuffle(images)
        return images
    
    def convertPassedTargetsToTrainingTargets(self,targets: list):
        '''
        convert the passed targets to the training targets
        exaple: [ [ 0 , 1 , 0 ] , [ 1 , 0 , 0 ] , [ 0 , 0 , 1 ] ] => [ 1 , 0 , 2 ]
        
        :param targets: the targets to convert
        :return: the converted targets
        '''
        # convert the passed targets to the training targets
        for i in range(len(targets)):
            targets[i] = targets[i].index(1)
        return targets
