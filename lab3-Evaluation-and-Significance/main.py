from eval import *
import numpy as np
from sklearn.metrics import accuracy_score
np.seterr(invalid='ignore')

print("Trues: " + "\t" + str(Y), "Top1: " + "\t" + str(allP[0]), "Top2: " + "\t" + str(allP[1]), "Top3" + "\t" + str(allP[2]), sep='\n')
print()
# makeManyCurves(Y, allP)
'''
Task A:
'''
print('Task A:')
print(
    "Accuracy of allP[0]: " + str(accuracy_score(Y, (allP[0] > .5).astype(int))),
    "Accuracy of allP[1]: " + str(accuracy_score(Y, (allP[1] > .5).astype(int))),
    "Accuracy of allP[2]: " + str(accuracy_score(Y, (allP[2] > .5).astype(int))),
    sep='\n'
)

print()
'''
Task B:
    * High threshold leads to high precision but low recall;
    * Low threshold leads to low precision but high recall;
    
    * F = 2*P*R/P+R - Harmonic mean (balanced f-measure or f-score)
    * Fb = (1+b^2)*P*R/b^2*P+R - weighted f-measure, where b => [0; inf), 
        => if b=1 : standard f-measure
        => if b=0 : focuses on recall
        => if b => inf : focuses on precision
'''

print('Task C:')
'''
Task C:
'''
print("The first place team is better than the second place team with 99.5% confidence =>", ttest(Y, allP[1], allP[0]))
print("The second place team is not significantly better than the third place team, even at 90% =>", ttest(Y, allP[2], allP[1]))
print("Run ttest on subset [0:500] =>", ttest(Y, allP[1], allP[0], restrictTo=500))
print("Run ttest on subset [0:750] =>", ttest(Y, allP[1], allP[0], restrictTo=500))

# Plotting
plt.plot([ttest(Y,allP[1],allP[0], i)[0] for i in range(10160)]) # 10160 => len of data
plt.show()

'''
Task D:
    * We can show it on drawn plot: 
        => 95% significance: can be shown when Y-axis (t-value) is >=1.64 and <=1.96, find the point and get X-axis (amount of data)
        => 99.5% significance: can be shown when Y-axis (t-value) is >=2.58, find the point and get X-axis (amount of data)
'''

'''
Task E:
    
'''