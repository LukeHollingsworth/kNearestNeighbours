import numpy as np
import matplotlib.pyplot as plt
import random


def GeneratePoints():
    N = 100
    S = np.zeros((N, 2))
    Y = np.zeros((N, 1))

    for n in range(N):
        x1 = np.random.uniform()
        x2 = np.random.uniform()
        y = np.random.randint(0, 2)
        s = (x1, x2)

        S[n] = s
        Y[n] = y

    return S, Y


def CalculateDistance(s, x1, x2):
    distance = np.sqrt((x1 - s[0])**2 + (x2 - s[1])**2)
    
    return distance


def ClosestNeighbors(S, x1, x2, k):
    distances = []
    neighbors = np.zeros((k, 2))

    for s in S:
        distance = CalculateDistance(s, x1, x2)
        distances.append(distance)

    indices = np.argsort(distances)[:k]

    for n, i in enumerate(indices):
        s = S[i]
        neighbors[n, :] = s

    return neighbors, indices


def GiveValue(S, x1, x2, Y, k):
    neighbors, indices = ClosestNeighbors(S, x1, x2, k)
    values = np.zeros((k, 1))

    for n, i in enumerate(indices):
        values[n] = Y[i]

    if sum(values) >= k/2:
        new_value = 1

    else:
        new_value = 0

    return new_value

def GenerateH(NTrain, S, Y, k):
    N = 100
    X1 = X2 = np.linspace(0, 1, N+1)
    X = []
    XSample = []
    h = []
    hSample = []

    for i in X1:
        for j in X2:
            newPoint = (i, j)
            X.append(newPoint)

            newValue = GiveValue(S, i, j, Y, k)
            h.append(newValue)

    indices = np.arange(0, N**2)
    sampleIndices = random.sample(list(indices), NTrain)
    
    for n in sampleIndices:
        XSample.append(X[n])
        hSample.append(h[n])

    return XSample, hSample


def GenerateHError(NTrain, S, Y, k):
    N = 100
    X1 = X2 = np.linspace(0, 1, N+1)
    X = []
    XSample = []
    h = []
    hSample = []
    coin = np.random.uniform()

    for i in X1:
        for j in X2:
            newPoint = (i, j)
            X.append(newPoint)

# Adding bias coin to determine whether y sampled from p_H or random
            if coin <= 0.8:
                newValue = GiveValue(S, i, j, Y, k)
            else:
                newValue = np.random.randint(0, 2)

            h.append(newValue)

    indices = np.arange(0, N**2)
    sampleIndices = random.sample(list(indices), NTrain)
    
    for n in sampleIndices:
        XSample.append(X[n])
        hSample.append(h[n])

    return XSample, hSample


def CalculateError(NTest, h, he):
    errors = []
    indices = np.arange(0, len(h))
    sampleIndices = random.sample(list(indices), NTest)

    for i in sampleIndices:
        error = abs(h[i] - he[i])
        errors.append(error)

    return sum(errors)

def PlotError(E):
    E, K = zip(*E)
    plt.scatter(K, E)
    plt.xlabel('k')
    plt.ylabel('Generalisation Error')
    plt.show()
    

def GenerateErrorPlot(NTrain, NTest):
    E = []
    K = np.arange(1, 50)

    for k in K:
        errorSum = 0

        for n in range(100):
            S, Y = GeneratePoints()
            h = GenerateH(NTrain, S, Y, k)[1]
            he = GenerateHError(NTrain, S, Y, k)[1]
            error = CalculateError(NTest, h, he)
            errorSum += error

        errorMean = errorSum / NTest
        print(errorMean, k)
        E.append((errorMean, k))

    PlotError(E)

    # Generalisation error linearly decreases with increasing k.
    # This is due to the fact that looking at more neighbouring values decreases the probability that a point generated randomly (and therefore incorrectly) will impact the value of the new point.


# def Test():
#
# 
#      
GenerateErrorPlot(NTrain=4000, NTest=1000)