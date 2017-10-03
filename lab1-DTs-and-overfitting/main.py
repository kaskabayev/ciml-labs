from data import *
from sklearn.tree import DecisionTreeClassifier
import time

X,Y,dictionary = loadTextDataBinary('text/sentiment.tr')
Xde,Yde,_ = loadTextDataBinary('text/sentiment.de', dictionary)
Xte,Yte,_ = loadTextDataBinary('text/sentiment.te', dictionary)

# print(X.shape)
# print(Y.shape)

# dt = DecisionTreeClassifier(max_depth=2)
# dt.fit(X, Y)
# showTree(dt, dictionary)

# Xde,Yde,_ = loadTextDataBinary('text/sentiment.de', dictionary)
# print(np.mean(dt.predict(Xde) == Yde))

print("Depth", "Training", "Development", "Test", sep="\t")
for i in range(1, 21):
    start = time.time()
    dt = DecisionTreeClassifier(max_depth=i)
    dt.fit(X, Y)
    end = time.time()
    # showTree(dt, dictionary)
    print(i, np.mean(dt.predict(X) == Y), np.mean(dt.predict(Xde) == Yde), np.mean(dt.predict(Xte) == Yte), sep="\t")