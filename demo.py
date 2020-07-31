from sklearn import datasets

# load a common datascience toy dataset, containing iris species observations
iris = datasets.load_iris()

# arrange data into attributes/feature matrix and a target/y matrix
x = iris.data
# this is the 1-dimensional array that contains the classes
#   that we are trying to predict
y = iris.target

# if you like, print the iris dataset information to add context
print(iris['DESCR'])
print('\n')

# you could also explore the pandas library as a way of examining the dataframe
import pandas as pd
irisdf = pd.DataFrame(x)
irisdf['class'] = y
print(irisdf.shape)
print(irisdf.head())

####### PREPARATION #########
from sklearn.model_selection import train_test_split

# arrange x, y matrices into training (80% of data) & testing (20% remaining)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

########################################################################################
########## DIY KNN ############

from knn import KNN

# instantiate knn model with 3 classes for our 3 iris species
knn = KNN(3)

# 'fit' the model to our 'known' training subsets
knn.fit(X_train, y_train)

# predict class for 'unseen' test subset matrix of features without classes
# while storing it for later comparison
our_prediction = knn.predict(X_test)

####### SKLEARN KNN #########

from sklearn.neighbors import KNeighborsClassifier

# our step 1:
# knn = KNN(3)
sklearn = KNeighborsClassifier(3)

# our step 2:
# knn.fit(X_train, y_train)
sklearn.fit(X_train, y_train)

# our step 3:
# our_prediction = knn.predict(X_test)
sklearn_prediction = sklearn.predict(X_test)

####### COMPARE ACCURACY #######

from sklearn.metrics import accuracy_score

print(f"accuracy of homebrewed knn: {accuracy_score(y_test, our_prediction)}\n")

print(f"accuracy of sklearn library results: {accuracy_score(y_test, sklearn_prediction)}\n")


