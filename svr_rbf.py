import numpy as np
import xlrd

loc = ("data.xlsx")
storyPoints = []
estimatedTime = []

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)

for i in range(sheet.nrows):
    storyPoints.append(sheet.cell_value(i, 1))

for i in range(sheet.nrows):
    estimatedTime.append(sheet.cell_value(i, 8))

print(storyPoints)
print(estimatedTime)
