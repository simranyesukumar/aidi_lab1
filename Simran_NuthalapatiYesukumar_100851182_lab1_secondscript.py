from sklearn.linear_model import LogisticRegression
#classify if an iris is virginica or not, then classify
#if it is versicolor or not, and finally classify if it is setosa or not.
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))

X = iris["data"][:, 3:] #petal width
#print(X)
y = (iris["target"] == 2).astype(np.int) #virginica

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression1 = LogisticRegression()
regression1.fit(X_train, y_train)
#print(regression1)

X_new = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
y_probability = regression1.predict_proba(X_new)

plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot(X_new, y_probability[:, 1], 'r-', label='virginica')
plt.plot(X_new, y_probability[:, 0], 'b--', label='Not virginica')
plt.xlabel("Petal Width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(fontsize=14)
plt.show()