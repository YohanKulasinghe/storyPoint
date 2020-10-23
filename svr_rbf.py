import numpy as np
import xlrd
import matplotlib.pyplot as plt
import pandas as pd

#x for story point 
#y for time in days

dataset=pd.read_csv("data.csv")
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
y=y.reshape(len(y),1)

print(x)
print(y)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
r=SVR(kernel="rbf")
r.fit(x, y)
print(sc_y.inverse_transform(r.predict(sc.transform([[100]]))))
 
plt.scatter(sc.inverse_transform(x),sc_y.inverse_transform(y), color="red")
plt.plot(sc.inverse_transform(x),sc_y.inverse_transform(r.predict(x)),color="blue")
plt.title("Salary Prediction (SVR) - RBF Kernal")
plt.xlabel("Story Point")
plt.ylabel("Time")
plt.show()