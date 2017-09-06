from mnist import *

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

prediction = sgd_clf.predict([some_digit])
#print(prediction)

#skfolds = StratifiedKFold(n_splits=3, random_state=42)
'''
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
'''
#cross = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#print(cross)
'''
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
'''
#neverscore = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#print(neverscore)

#y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
#y_train_pred = cross_val_predict(never_5_clf, X_train, y_train_5, cv=3)


#cm = confusion_matrix(y_train_5, y_train_pred)
#print(cm)

#precision = precision_score(y_train_5, y_train_pred)
#print(precision)
#recall = recall_score(y_train_5, y_train_pred) # == 4344 / (4344 + 1077)
#print(recall)

#y_scores = sgd_clf.decision_function([some_digit])
#print(y_scores)

#threshold = 0
#y_some_digit_pred = (y_scores > threshold)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

print('y_scores')
print(y_scores)

#precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores[:, 1])

#def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
#    plt.xlabel("Threshold")
#    plt.legend(loc="upper left")
#    plt.ylim([0, 1])

#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#plt.show()

#fpr, tpr, thresholds = roc_curve(y_train_5, y_scores[:, 1])

#plot_roc_curve(fpr, tpr)
#plt.show()

print(roc_auc_score(y_train_5, y_scores[:,1]))