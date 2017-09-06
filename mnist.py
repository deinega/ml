from sklearn.datasets import fetch_mldata
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.base import clone
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import SGDClassifier

print('downloading...')
mnist = fetch_mldata('MNIST original')
print('done')
#print(mnist)

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

#print(shuffle_index)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#print(y_train_5)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
