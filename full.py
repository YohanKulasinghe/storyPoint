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
r=SVR(kernel='rbf', C = 0.78121, tol = 0.05 )
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
    print(abs(testSetActualTime[i]-testSetPredictedTime[i])/testSetActualTime[i])
    MMRE = MMRE + (abs(testSetActualTime[i]-testSetPredictedTime[i])/testSetActualTime[i])

MMRE = MMRE/n * 100

print("\nMMRE For time= %f " % (MMRE))

# temp1 = 0

# for i in range(n):
#     temp1 = temp1 + abs(testSetActualTime[i]-testSetPredictedTime[i])

# print(temp1)
# PRED = 1 - (temp1 / n) * 100
# print(PRED)

team_salary=560679
non_tech_salary=183451
equipment=34821
depreciation=8736
rent=14634
travelling=38279
furniture=2356
utilityBills=27541
coppyright=15239
software_purchase=12781
repair_and_maintenance=8393
sanitary=5782
marketing=4782
other=24790

fraction_non_tech_salary= non_tech_salary / team_salary
fraction_equipment=equipment / team_salary
fraction_depreciation=depreciation / team_salary
fraction_rent=rent / team_salary
fraction_travelling=travelling / team_salary
fraction_furniture=furniture / team_salary
fraction_utilityBills=utilityBills / team_salary
fraction_coppyright=coppyright / team_salary
fraction_software_purchase=software_purchase / team_salary
fraction_repair_and_maintenance=repair_and_maintenance / team_salary
fraction_sanitary=sanitary / team_salary
fraction_marketing=marketing / team_salary
fraction_other= other / team_salary

cost_driver = 1 + fraction_non_tech_salary + fraction_equipment + fraction_depreciation + \
    fraction_rent + fraction_travelling + fraction_furniture + \
    fraction_utilityBills + fraction_coppyright + fraction_software_purchase + \
    fraction_repair_and_maintenance + fraction_sanitary + \
    fraction_marketing + fraction_other

print(cost_driver)

predicted_cost = cost_driver * 230000 * 58/30

print(predicted_cost)

