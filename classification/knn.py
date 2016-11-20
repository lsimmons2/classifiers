
from scipy.spatial import distance
from sklearn.metrics import classification_report

def euc(a, b):
    return distance.euclidean(a,b)


class MyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def closest(self, X_test):
        closest = euc(X_test, self.X_train[0])
        closest_index = 0
        for i in range (1, len(self.X_train)):
            if closest > euc(self.X_train[i], X_test):
                closest = euc(self.X_train[i], X_test)
                closest_index = i
        return self.y_train[closest_index]


    def predict(self, features):
        predictions = []
        for feature in features:
            predictions.append(self.closest(feature))
        return predictions
