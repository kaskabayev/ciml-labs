from imports import *

h = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=1))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)

print(mean(P == WineData.Yte))
print(mode(WineData.Y))
print(WineData.labels[1])
print(mean(WineData.Yte == 1))

P = h.predictAll(WineData.Xte, useZeroOne=True)
print(mean(P == WineData.Yte))