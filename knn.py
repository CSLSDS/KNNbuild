import pandas as pd
import numpy as np
import statistics as stat

class KNN:
    X_train = None
    y_train = None
    def __init__(self, K):
        # K Nearest Neighbors requires a value for K,
        #   which will be the number of classifications/groups created
        self.K = K

    def fit(self, X_train, y_train):
        # fit method requires data as input,
        #   arranged in features (X_train) and target (y_train) matrices
        self.X_train = X_train
        self.y_train = y_train

    def edist(self, rowA, rowB):
        # calculates Euclidean distance between points
        return np.linalg.norm(rowA-rowB)

    def predict(self, val_or_test):
        # takes as input a validation or test feature matrix (X_val/X_test)
        # Here, euclidean distances will be calculated,
        #   classifying points based on minimum distance (nearest neigbors)

        # initialize list to return predictions
        predictions = []
        for secondary_obs in val_or_test:
        # iterate over observations in secondary test/validation dataset
            # store in initialized list to sort/compare distances/neighbors
            distances = []
            for ix, train_obs in enumerate(self.X_train):
            # iterate over index, observations in original training set
                # calculate euclidean distance between train obs and test obs
                distance = self.edist(train_obs, secondary_obs)
                # [distance, index] appended to distances for comparison
                distances.append([distance, ix]) # Ksubset entries
            ranked_neighbors = sorted(distances)
            # subset ranked_neighbors to length/number defined in K
            Ksubset = ranked_neighbors[:self.K]

            # assign the index matching target training observation
            #    for the top ranked neighbors (corresponding classes)
            predict = [self.y_train[ix[1]] for ix in Ksubset]
            
            # return prediction with max count/"votes" as the 
            #   nearest neigbhor/most likely class
            result = max(predict, key = predict.count)
            
            predictions.append(result)
        
        return predictions