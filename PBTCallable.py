#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 19:52:28 2020

@author: zixiangpei
"""

import PBTToyExample as exampleCode
import random
import numpy as np

class stepClass():
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, theta, h):
        return exampleCode.step(theta, h, self.alpha)


class readyClass():
    def __init__(self, threshold):
        self.threshold = threshold
        
    def __call__(self, p, t, P):
        return exampleCode.ready(p, t, P, self.threshold)
    
class sampleExploitClass():
    def __call__(self, P, n):
        obj = random.sample(list(P), n)[0]
        return(obj)

class exploitClass():
    def __init__(self, sampleExploit):
        self.sampleExploit = sampleExploit
    
    def __call__(self, h, theta, p, P):
        return(exampleCode.exploit(h, theta, p, P, self.sampleExploit))
    
class sampleExploreClass():
    def __call__(self,n):
        return(np.random.uniform(0.8, 1.2, n))

class exploreClass():
    def __init__(self, sampleExplore):
        self.sampleExplore = sampleExplore
    
    def __call__(self, hPrime, thetaPrime, P):
        return(exampleCode.explore(hPrime, thetaPrime, P, self.sampleExplore))
    
class endofTrainClass():
    def __init__(self, numIt):
        self.numIt = numIt
        
    def __call__(self, t):
        return(t >= self.numIt)
        
        
        
    
    