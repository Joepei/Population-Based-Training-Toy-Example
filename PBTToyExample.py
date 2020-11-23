import numpy as np
import random

#Toy example data format:
#theta: {"theta": [theta0, theta1], "h": [h0, h1], "id": id}
#h: [h0, h1]
#p: {p: p, id: id}
#t: t
#P: [{theta: theta, h: h, p: p, t: t}, {theta: theta, h: h, p: p, t: t}]

#Using gradient descent to minimize -Q
def step(theta, h, alpha): #need to make the function convex in order to apply gradient descent so instead of maximizing Q, minimize -Q.
    newtheta = theta.copy()
    newtheta["theta"][0] -= alpha*(2*h[0]*newtheta["theta"][0])
    newtheta["theta"][1] -= alpha*(2*h[1]*newtheta["theta"][1])
    return(newtheta)

#Evaluating the parameters with Q Hat
def eval(theta):
    pValue = 1.2-(theta["theta"][0]**2 + theta["theta"][1]**2)
    p = {}
    p["p"] = pValue
    p["id"] = theta["id"]
    return(p)

#Checks whether the change of value with gradient descent is still effective
def ready(p,t,P,threshold):
    return(t % 5 == 4)

#Randomly draws a competitior in the population except the worker itself to compete with, and the winner's parameters gets used in the next iteration
def exploit(h, theta, p, P, sampleFunction): #made a change here to not use sampleFunction
    opponent = sampleFunction(P, 1)
    while opponent["p"]["id"] == p["id"]: #to ensure that opponent is different from the worker itself
        opponent = random.sample(list(P),1)[0]
    if opponent["p"]["p"] >= p["p"]:
        thetaPrime = {}
        thetaPrime["theta"] = opponent["theta"]["theta"][:]
        thetaPrime["h"] = h[:]
        thetaPrime["id"] = theta["id"]
        return (h, thetaPrime)
    else:
        return (h, theta)

#Randomly draws a perturbation rate in the distribution and changes the (hyper)parameters by the perturbation rate
def explore(hPrime, thetaPrime, P, sampleFunction):
    h0Perturb, h1Perturb = sampleFunction(2)
    newh = hPrime[:]
    newh[0] = (hPrime[0]+0.1) * h0Perturb
    newh[1] = (hPrime[1]+0.1) * h1Perturb
    newtheta = thetaPrime.copy()
    newtheta["h"] = newh.copy()
    return(newh, newtheta)

#Updates Population with newly calculated results
def update(P, theta, h, p, t):
    for counter, item in enumerate(P):
        if item["p"]["id"] == p["id"]:
            P[counter]["theta"] = theta.copy()
            P[counter]["h"] = h[:]
            P[counter]["p"] = p.copy()
            P[counter]["t"] = t+1
    return(P, t+1)

#Returns the theta with highest p in the population
def thetaHighestp(P):
    max = -np.inf
    for item in P:
        if item["p"]["p"] > max:
            max = item["p"]["p"]
            maxItem = item
    return(maxItem["theta"])

#parses value into the four parameters required in the algo
def parseValue(value):
    theta = value["theta"]
    h = value["h"]
    p = value["p"]
    t = value["t"]
    return(theta, h, p, t)




    

