"""
Test SGD implementation with Logistic Regression and Iris dataset.
"""

from archive.logreg_old import *
from old import *

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split    

# Load the Iris dataset
iris = load_iris()

# Separate features and target labels
X = iris.data
y = iris.target

# Encode target labels (0 for Setosa, 1 for non-Setosa)
y = (y == 0).astype(int)  # 0 for Setosa, 1 for others

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
optimizer = SGD()

model.fit(X_train, y_train, optimizer, step_size=1, update_method='avgEachIter')

y_pred = model.predict(X_test)
print(y_test)
print(y_pred)