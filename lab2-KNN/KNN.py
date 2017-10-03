import sys,os
from math import *

def loadDigitData(filename, maxExamples=100000):
    h = open(filename, 'r')
    D = []
    for l in h.readlines():
        a = l.split()

        if len(a) > 1:
            y = float(a[0])
            if y > 0.5: y = 1.
            else: y = -1.
            x = {}
            for i in range(1, len(a)):
                v = float(a[i]) / 255.
                if v > 0.:
                    print(i, v)
                    x[i] = v
            D.append( (x,y) )
            if len(D) >= maxExamples:
                break
    h.close()
    return D

def exampleDistance(x1, x2): # x1 is training text row, x2 is test text row
    dist = 0.
    for i,v1 in x1.items(): # iterates over training row of points
        v2 = 0.
        if i in x2: # if training point is has the same example id as test points
            v2 = x2[i]
        dist += (v1 - v2) * (v1 - v2) # calculate the difference of 2 points (v1 training point, v2 test point) and square it
                                      # and sum with distance of other points
    for i,v2 in x2.items(): # iterates over test row of points
        if i not in x1: # if point is not in training points
            dist += v2 * v2 # calculate the square of same point and sum it with distance
    return sqrt(dist) # returns distance

# returns list of K (dist, n) where n is the nth training example
def findKNN(D, xhat, K):  # D is the training text, xhat is the test point
    allDist = []
    for n in range(len(D)):
        (x,y) = D[n]
        allDist.append( (exampleDistance(x, xhat), n) )
    allDist.sort()
    return allDist[0:K]

def classifyKNN(D, knn, K = 0.000000000000000000000000000000000000000000000001):
    yhat = 0
    for (dist,n) in knn: # iterates over knn distances and point
        (x,y) = D[n] # gets point of training set by point in knn
        yhat = yhat + y
        # yhat = exp(-dist / K)

    if yhat > 0.:
        return 1.
    else:
        return -1.

def computeErrorRate(trainingData, testData, allK):
    maxK = allK[0]
    err = []

    for k in allK:
        if k > maxK: maxK = k; # defines maximum K
        err.append(0.)

    for (x,y) in testData:
        knn = findKNN(trainingData, x, maxK) # gets KNN by quantity of maxK
        for i in range(len(allK)):
            yhat = classifyKNN(trainingData, knn[0:allK[i]], allK[i]) # strips number of the nearest points by K
            if y * yhat < 0:
                err[i] += 1.

    for i in range(len(allK)):
        err[i] /= float(len(testData))

    return err


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('usage: python KNN.py [training filename] [testing filename] [K1] [K2] ... [Klast]')
        exit(-1)

    tr = loadDigitData(sys.argv[1])
    te = loadDigitData(sys.argv[2], 100)
    allK = [int(arg) for arg in sys.argv[3:]]
    print("\t   ".join([str(err) for err in computeErrorRate(tr, te, allK)]))