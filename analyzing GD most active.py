from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
import pandas as pd

data = pd.read_csv("GD Most Active 2.csv")
X = data.iloc[:,:19].values
y = data.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
for i in range(0,len(X[0]),1):
	X[:,i] = le.fit_transform(X[:,i])

#------------RFR-------------#
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)

t = pd.read_csv("test2.csv")
Z = t.iloc[:,:].values
print(len(X[0]))
print(len(Z[0]))
for i in range(0,19,1):
	Z[:,i] = le.fit_transform(Z[:,i])


predictions=rf.predict(X)
plt.scatter(y, predictions)
plt.show()


pr=rf.predict(Z)
Z1 = Z[:,0]
print(Z1)
plt.plot(Z1, pr)
plt.show()
