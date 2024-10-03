import utilities.getClasses as getClasses
import random
import os
import numpy as np
import classes.FeatureExtractor as fe

class DataSetManager:
    def __init__(self):
        self.imagesPath = "imgs/"
        self.jsonPath = "imgs\dataSet.json"
    
    '''
    get the entire dataset in a form of a list of image paths
    '''
    def getAllImages(self):
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
    
    '''
    given the path of the image, return the correct prediction of the image that ambrogio should return
    '''
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
    
    '''
    get a random image path of the dataset
    '''
    def getRandomImage(self):
        # go in the imgs folder and get a random image from a random class
        classes = getClasses.getClasses()
        randomClass = random.choice(classes)
        pathChosen = self.imagesPath + randomClass + "/"
        images = os.listdir(pathChosen)
        
        return pathChosen + random.choice(images)        
    
    '''
    The data will be a tuple
    return 0 => the training set, 1 => the convalidation set and 2 => the test set
    '''
    def partitionDataSet(self):
        # partition the data set into training and test set
        images = self.getAllImages()
        random.shuffle(images)
        trainingSet = images[:int(len(images)*0.3)]
        convalidationSet = images[int(len(images)*0.3):int(len(images)*0.6)]
        testSet = images[int(len(images)*0.6):]
        
        
        return trainingSet,convalidationSet, testSet
    
    
    def create_minibatches(self,X, y, batch_size):
        # check if X and y are numpy arrays
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)


        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Dividi in batch
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

    def create_minibatches_from_scratch(self, batch_size):
        featureExtractor = fe.FeatureExtractor()
        X = self.getAllImages()
        y = [self.getCorrentPredictionOfImage(image) for image in X]
        X = [featureExtractor.extract_features(path) for path in X]
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)


        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Dividi in batch
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]
    
    def randomShuffleDataSet(self):
        # shuffle the dataset
        images = self.getAllImages()
        random.shuffle(images)
        return images
