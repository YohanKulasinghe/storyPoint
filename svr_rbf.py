from typing import AbstractSet
import numpy as np
import xlrd
import matplotlib.pyplot as plt
import pandas as pd

#x for story point 
#y for time in days

dataset=pd.read_csv("data.csv")
print(dataset)
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,2:].values

spArray = x.flatten()
timeArray = y.flatten()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)


y = y.ravel()
from sklearn.svm import SVR
r=SVR(kernel="rbf")
r.fit(x,y)


plt.scatter(sc.inverse_transform(x),sc_y.inverse_transform(y), color="red")
plt.plot(sc.inverse_transform(x),sc_y.inverse_transform(r.predict(x)),color="blue")
plt.title("Salary Prediction (SVR) - RBF Kernal")
plt.xlabel("Story Point")
plt.ylabel("Time")
#plt.show()

predictedTime = []
time = []

for sp in spArray :
    x = sc_y.inverse_transform(r.predict(sc.transform([[sp]])))
    predictedTime.append(round(x[0]))

for t in timeArray :
    time.append(t)    

print(time)
print(predictedTime)

n = 21

MMRE = 0

for i in range(n):
    MMRE = MMRE + (abs(time[i]-predictedTime[i]))/time[i]

print("\nMMRE = %f" % (MMRE))

# tempP = 0

# for i in range(n):
#     tempP = tempP + abs(time[i]-predictedTime[i])
#     print(abs(time[i]-predictedTime[i]))
#     print(tempP)

# PRED = (1 - (tempP / n)) * 100

# print("PRED = %f" % (PRED))




#print(sc_y.inverse_transform(r.predict(sc.transform([[74]]))))