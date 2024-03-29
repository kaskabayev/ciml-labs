from math import *
import random
from random import randint
from numpy import *
import matplotlib.pyplot as plt
from KNN import subsampleExampleDistance

waitForEnter=False

def generateUniformExample(numDim):
    return [random.random() for _ in range(numDim)]

def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for _ in range(numEx)]

def computeExampleDistance(x1, x2):
    dist = 0.0
    for d in range(len(x1)):
        dist += (x1[d] - x2[d]) * (x1[d] - x2[d])
    return sqrt(dist)

def computeDistances(data):
    N = len(data)
    D = len(data[0])
    DP = randint(1, D)

    dist = []
    for n in range(N):
        for m in range(n):
            dist.append( computeExampleDistance(data[n],data[m]) / sqrt(D))
            # dist.append( subsampleExampleDistance(data[n],data[m], DP) / sqrt(DP))
    return dist

N    = 200                   # number of examples
Dims = [2, 8, 32, 128, 512, 784]   # dimensionalities to try
Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF', '#DDDD12']
Bins = arange(0, 1, 0.02)

plt.xlabel('distance / sqrt(dimensionality)')
plt.ylabel('# of pairs of points at that distance')
plt.title('dimensionality versus uniform point distances')

for i,d in enumerate(Dims):
    distances = computeDistances(generateUniformDataset(d, N))
    print("D=%d, average distance=%g" % (d, mean(distances) * sqrt(d)))
    plt.hist(distances,
             Bins,
             histtype='step',
             color=Cols[i])
    if waitForEnter:
        plt.legend(['%d dims' % d for d in Dims])
        plt.show(False)
        x = input('Press enter to continue...')

plt.legend(['%d dims' % d for d in Dims])
plt.savefig('fig.pdf')
plt.show()