from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

datasets = pd.read_csv('Housing_Data.csv')

X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 2, random_state=40)

lr = LinearRegression()
lr = lr.fit(X_train, y_train)

Y_Pred = lr.predict(X_test)

fig, axes = plt.subplots(1, 5, figsize=(13, 4))
for i in range(5):
    axes[i].scatter(X_train[:, i], y_train[:], marker='^', color='r', label="Train set", s=60)
    axes[i].scatter(X_test[:, i], y_test[:], marker='^', color='b', label="Test set", s=60)
    axes[i].scatter(X_test[:, i], Y_Pred[:], marker='^', color='k', label="Prediction set", s=60)
    axes[i].set_xlabel(str(i) + "-th Feature")
    axes[i].legend(loc='upper left')
    axes[i].set_title(str(i) + "-th Feature")
axes[0].set_ylabel("Price")
plt.show()
