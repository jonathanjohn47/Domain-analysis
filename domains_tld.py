from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
import pandas as pd

data = pd.read_csv("GD Most Active 2.csv")
X = data.iloc[:,[2,10,11]].values
y = data.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
for i in range(0,len(X[0]),1):
	X[:,i] = le.fit_transform(X[:,i])

#------------RFR-------------#
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 400, random_state = 1)
rf.fit(X,y)

t = pd.read_csv("test2.csv")
Z = t.iloc[:,[2,9,10]].values
'''
for i in range(0,len(Z[0]),1):
	Z[:,i] = le.fit_transform(X[:,i])
'''
predictions=rf.predict(X)
plt.scatter(y, predictions)
plt.show()

pr=rf.predict(Z)

A = range(0,len(Z), 1)
plt.plot(A,pr)
plt.show()

a = []
bid = input("Enter the no. of bids you want to check: ")
bids = int(bid)
for i in range(0, len(Z), 1):
	if(pr[i]>bids):
		a.append(i+1)
	else:
		pass
print(a)
print("\n")
print(str(len(a)), "domains found with estimated bids more than " + str(bids))
print(Z[a,:])


