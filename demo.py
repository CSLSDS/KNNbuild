from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn import KNN
from sklearn import datasets

# load a common datascience toy dataset, containing iris species observations
iris = datasets.load_iris()

# arrange data into attributes/feature matrix and a target/y matrix
x = iris.data
# this is the 1-dimensional array that contains the classes
#   that we are trying to predict
y = iris.target

# print the iris dataset information to add context
print(iris['DESCR'])
print('\n')

# arrange x, y matrices into training (80% of data) & testing (20% remaining)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# instantiate knn model with 3 classes
knn = KNN(3)

# 'fit' the model to our 'known' training subsets
knn.fit(X_train, y_train)

# predict class for 'unseen' test subset matrix of features without classes
y_pred = knn.predict(X_test)

# check and print accuracy of the constructed predictive modeling algorithm
print(f"accuracy of homebrewed knn: {accuracy_score(y_test, y_pred)}\n")

# import sklearn's implementation and repeat for comparison
from sklearn.neighbors import KNeighborsClassifier
sklearn = KNeighborsClassifier(3)
sklearn.fit(X_train, y_train)
pred = sklearn.predict(X_test)
print(f"accuracy of sklearn library results: {accuracy_score(y_test, pred)}\n")