from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

'''

'''
def get_hyperplane(X_train, y_train, X_test, y_test):
    svm_classifier = SVC(kernel='linear')

    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

