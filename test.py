from sklearn import datasets
from sklearn.metrics import classification_report

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

from classification import knn



iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = .5)



my_knn = knn.MyKNN()
skl_knn = KNeighborsClassifier()
lin_reg = linear_model.LinearRegression()

my_knn.fit(X_train, y_train)
skl_knn.fit(X_train, y_train)
lin_reg.fit(X_train, y_train)



my_predictions = my_knn.predict(X_test)
skl_predictions = skl_knn.predict(X_test)



print 'my_knn:'
print classification_report(y_test, my_predictions)

print 'skl_knn:'
print classification_report(y_test, skl_predictions)

print 'lin_reg.coef_'
print lin_reg.coef_
