from imports import *

## Normal dataset
# h = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=1))
# h.train(WineData.X, WineData.Y)
# P = h.predictAll(WineData.Xte)

# print(mean(P == WineData.Yte))
# print(mode(WineData.Y))
# print(WineData.labels[1])
# print(mean(WineData.Yte == 1))

# P = h.predictAll(WineData.Xte, useZeroOne=True)
# print(mean(P == WineData.Yte))

## Smaller dataset
# h = multiclass.OAA(5, lambda: DecisionTreeClassifier(max_depth=3))
# h.train(WineDataSmall.X, WineDataSmall.Y)
# P = h.predictAll(WineDataSmall.Xte)

# print(mean(P == WineDataSmall.Yte))
# print(mean(WineDataSmall.Yte == 1))

# Sauvignon-Blanc
# print(WineDataSmall.labels[0])
# print(util.showTree(h.f[0], WineDataSmall.words))

'''
Task A:
    1. For Sauvignon-Blanc most indicative words are: citrus, lime, gooseberry
    2. For NOT Sauvignon-Blanc most indicative words are: extremely, flavors
'''

# Pinot-Noir
# print(WineDataSmall.labels[2])
# print(util.showTree(h.f[2], WineDataSmall.words))

'''
Task A:
    3. For Pinot-Noir most indicative words are: strawberry, allspice, cassis
    4. For NOT Pinot-Noir most indicative words are: verdot, nice
'''

'''
Task B:
    
'''
start = time.clock()
h = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)
end = time.clock()

print ("Time used: %.2gs" % (end-start))

print(WineData.labels[17])
print(util.showTree(h.f[17], WineData.words))