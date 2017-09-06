from mnist import *

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)

y_multilabel = np.c_[y_train_large, y_train_odd]

#print(y_multilabel)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

prediction = knn_clf.predict([some_digit])
#print(prediction)

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1 = f1_score(y_train, y_train_knn_pred, average="macro")
print(f1)