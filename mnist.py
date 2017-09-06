from sklearn.datasets import fetch_mldata
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.base import clone
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import SGDClassifier

print('downloading...')
mnist = fetch_mldata('MNIST original')
print('done')
print(mnist)

#print(mnist.data.shape)

X, y = mnist["data"], mnist["target"]

#print(type(mnist))
#print(type(X))

some_digit = X[36000]
#some_digit_image = some_digit.reshape(28, 28)
#plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
#plt.axis("off")
#plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#print(y_train[0])
#print(y_train[1])
np.random.seed(0)
shuffle_index = np.random.permutation(60000)

print(shuffle_index)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

print(y_train_5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

prediction = sgd_clf.predict([some_digit])
print(prediction)

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495

cross = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(cross)

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
neverscore = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(neverscore)