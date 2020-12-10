from typing import AbstractSet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

trainDataset=pd.read_csv("data.csv")
print("\nPrinting Trainig Dataset\n")
print(trainDataset)
trainSP=trainDataset.iloc[:,1:-1].values
trainT=trainDataset.iloc[:,2:].values

sc=StandardScaler()
trainSP=sc.fit_transform(trainSP)
sc_y=StandardScaler()
trainT=sc_y.fit_transform(trainT)

trainT = trainT.ravel()
from sklearn.svm import SVR
r=SVR(kernel="rbf")
r.fit(trainSP,trainT)

plt.scatter(sc.inverse_transform(trainSP),sc_y.inverse_transform(trainT), color="red")
plt.plot(sc.inverse_transform(trainSP),sc_y.inverse_transform(r.predict(trainSP)),color="blue")
plt.title("Salary Prediction (SVR) - RBF Kernal")
plt.xlabel("Story Point")
plt.ylabel("Time")
plt.show()

trainDataset=pd.read_csv("evaluate.csv")
print("\n\n Printing Evaluation Dataset\n")
print(trainDataset)
testSP=trainDataset.iloc[:,1:-1].values
testT=trainDataset.iloc[:,2:].values

testSPList = testSP.flatten()
testTimeList = testT.flatten()

testSetActualTime = []
testSetPredictedTime = []


for sp in testSPList :
    x = sc_y.inverse_transform(r.predict(sc.transform([[sp]])))
    testSetPredictedTime.append(round(x[0]))

for t in testTimeList :
    testSetActualTime.append(t)  

print("\n Output Summery\n")

print("Actual ", testSetActualTime) 
print("Predct ", testSetPredictedTime)

n = len(testSetActualTime)

MMRE = 0

for i in range(n):
    MMRE = MMRE + (abs(testSetActualTime[i]-testSetPredictedTime[i]))/testSetActualTime[i]

print("\nMMRE = %f" % (MMRE))
