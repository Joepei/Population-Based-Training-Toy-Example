import numpy as np
import unittest
from ddt import ddt, data, unpack
import PBTCallable as callableCode
import PBTToyExample as targetCode

@ddt
class TestToyExample(unittest.TestCase):
    def setUp(self):
        self.P = [{"theta": {"theta": [0.9, 0.9],"h": [1,0], "id": 1}, "h": [1,0], "p": {"p": -0.42, "id": 1}, "t": 0 }, {"theta": {"theta": [0.9, 0.9],"h": [0,1],"id": 2}, "h": [0,1], "p": {"p": -0.42,"id": 2}, "t": 0}, {"theta": {"theta": [-0.9,-0.9], "h":[0.5,0.3], "id": 3}, "h": [0.5,0.3], "p": {"p": -0.42, "id": 3}, "t": 0 }, {"theta": {"theta": [-0.9, -0.9], "h": [0.3,0.5],"id": 4}, "h": [0.3,0.5], "p": {"p": -0.42, "id": 4}, "t": 0 }, {"theta": {"theta": [-0.2, 0.4],"h": [0.7,0.9],"id": 5}, "h": [0.7,0.9], "p": {"p": 1,"id": 5}, "t": 0}, {"theta": {"theta": [0, 0],"h": [0.5,0.5], "id": 6}, "h": [0.5,0.5], "p": {"p":1.2, "id": 6}, "t": 0}, {"theta": {"theta": [0.6, -1.1],"h": [1.3,1.5],"id": 7}, "h": [1.3,1.5], "p": {"p": -0.37, "id": 7}, "t": 0 }, {"theta": {"theta": [0, 0.4], "h": [0.4,0], "id": 8}, "h": [0.4,0], "p": {"p": 1.04, "id": 8}, "t": 0}, {"theta": {"theta": [0.75, -0.15], "h": [0.9,1.9], "id": 9}, "h": [0.9,1.9], "p": {"p": 0.615,"id": 9}, "t": 0}, {"theta": {"theta": [-0.23, -0.76],"h": [2.4,0.35],"id": 10}, "h": [2.4,0.35], "p": {"p": 0.5695, "id": 10}, "t": 0}]
        
        #Below are dummy functions used to sample explore and exploit parameters, in actual implementation a random sampling method will be provided.
        self.exploitSample1 = lambda x,y: {"theta": {"theta": [-0.23, -0.76],"h": [2.4,0.35],"id": 10}, "h": [2.4,0.35], "p": {"p": 0.5695, "id": 10}, "t": 0}
        self.exploitSample2 = lambda x,y: {"theta": {"theta": [0.9, 0],"h": [1,0], "id": 1}, "h": [1,0], "p": {"p": 0.39, "id": 1}, "t": 0 }
        self.exploitSample3 = lambda x,y: {"theta": {"theta": [-0.2, 0.4],"h": [0.7,0.9],"id": 5}, "h": [0.7,0.9], "p": {"p": 1,"id": 5}, "t": 0}
        self.exploreSample1 = lambda n: (0.1, 0.2)
        self.exploreSample2 = lambda n: (0, 0)
        self.exploreSample3 = lambda n: (-0.4, -0.3)
        self.distribution = [] #dummy variable
    
    #The next four functions tests are upon step() function
    @data(({"theta": [0.9, 0.9], "h": [1,0], "id": 1}, [1,0], 0.1, [0.72,0.9]), ({"theta": [0.4, 0.8],"h": [1.2,1.5],"id": 2}, [1.2,1.5], 0.18, [0.2272, 0.368]))
    @unpack
    def testStepPositive(self, theta, h, alpha, expectedResult):
        step =  callableCode.stepClass(alpha)
        updated_Theta = step(theta, h)
        first = updated_Theta["theta"][0]
        second = updated_Theta["theta"][1]
        self.assertListEqual(theta["h"], updated_Theta["h"])
        self.assertEqual(theta["id"], updated_Theta["id"])
        self.assertAlmostEqual(first,expectedResult[0],places=5)
        self.assertAlmostEqual(second,expectedResult[1],places=5)
    
    @data(({"theta": [0,0], "h": [0.7,0.8], "id": 2}, [0.7, 0.8], 0.2, [0,0]), ({"theta": [0,1], "h": [0.5,0.3], "id": 2}, [0.5, 0.3], 0.2, [0,0.88]), ({"theta": [1,0], "h": [1.4, 0.9], "id": 2}, [1.4, 0.9], 0.3, [0.16,0]))
    @unpack
    def testStepZero(self, theta, h, alpha, expectedResult):
        step =  callableCode.stepClass(alpha)
        updated_Theta = step(theta, h)
        first = updated_Theta["theta"][0]
        second = updated_Theta["theta"][1]
        self.assertListEqual(theta["h"], updated_Theta["h"])
        self.assertEqual(theta["id"], updated_Theta["id"])
        self.assertAlmostEqual(first,expectedResult[0],places=5)
        self.assertAlmostEqual(second,expectedResult[1],places=5)
    
          
    @data(({"theta": [-0.9,-0.9], "h": [0.5,0.3], "id": 3}, [0.5, 0.3], 0.25, [-0.675, -0.765]), ({"theta": [-2.5, -4],"h": [0.4, 0.6], "id": 10},[0.4,0.6],0.17, [-2.16,-3.184]),({"theta": [-0.25,-1.1],"h": [0.5,0.5], "id": 14}, [0.5,0.5],0.05,[-0.2375, -1.045]))
    @unpack
    def testStepNegative(self, theta, h, alpha, expectedResult):
        step = callableCode.stepClass(alpha)
        updated_Theta = step(theta,h)
        first = updated_Theta["theta"][0]
        second = updated_Theta["theta"][1]
        self.assertListEqual(theta["h"], updated_Theta["h"])
        self.assertEqual(theta["id"], updated_Theta["id"])
        self.assertAlmostEqual(first,expectedResult[0],places=5)
        self.assertAlmostEqual(second,expectedResult[1],places=5)
    
    @data(({"theta": [0.6, -1.1], "h": [1.3,1.5], "id": 7}, [1.3,1.5], 0.13, [0.3972, -0.671]), ({"theta": [-0.2, 0.4], "h": [0.7,0.9],"id": 5}, [0.7,0.9], 0.21, [-0.1412, 0.2488]))
    @unpack
    def testStepMixed(self, theta, h, alpha, expectedResult):
        step = callableCode.stepClass(alpha)
        updated_Theta = step(theta, h)
        first = updated_Theta["theta"][0]
        second = updated_Theta["theta"][1]
        self.assertListEqual(theta["h"], updated_Theta["h"])
        self.assertEqual(theta["id"], updated_Theta["id"])
        self.assertAlmostEqual(first,expectedResult[0],places=5)
        self.assertAlmostEqual(second,expectedResult[1],places=5)
    
    #The next four tests are upon eval() function
    @data(({"theta": [0.9, 0.9], "h": [1,0], "id": 1}, {"p": -0.42, "id": 1}), ({"theta": [1.2,0.8], "h": [0.3,0.7], "id": 15}, {"p": -0.88, "id": 15}), ({"theta": [2.1, 1.7], "h": [0.2, 0.9], "id": 20}, {"p": -6.1, "id": 20}))
    @unpack
    def testEvalPositive(self, theta, expectedResult):
        result = targetCode.eval(theta)
        self.assertAlmostEqual(result["p"], expectedResult["p"], places = 5)
        self.assertEqual(result["id"], expectedResult["id"])
    
    @data(({"theta":[0, 0], "h": [0.5,0.5], "id": 6} , {"p": 1.2, "id": 6}), ({"theta": [0, 0.4], "h": [0.4,0], "id": 8}, {"p": 1.04, "id": 8}), ({"theta": [0.9, 0], "h": [0.2, 0.3], "id": 16},{"p": 0.39, "id": 16}))
    @unpack
    def testEvalZero(self, theta, expectedResult):
        result = targetCode.eval(theta)
        self.assertAlmostEqual(result["p"], expectedResult["p"], places = 5)
        self.assertEqual(result["id"], expectedResult["id"])
    
    @data(({"theta": [-0.23, -0.76], "h": [2.4,0.35], "id": 10}, {"p": 0.5695, "id": 10}), ({"theta": [-0.9, -0.9], "h": [0.3,0.5], "id": 4}, {"p":-0.42, "id": 4}))
    @unpack
    def testEvalNegative(self, theta, expectedResult):
        result = targetCode.eval(theta)
        self.assertAlmostEqual(result["p"], expectedResult["p"], places = 5)
        self.assertEqual(result["id"], expectedResult["id"])
    
    @data(({"theta": [0.6, -1.1], "h": [1.3,1.5], "id": 7}, {"p": -0.37, "id": 7}), ({"theta": [0.75, -0.15], "h": [0.9,1.9], "id": 9}, {"p": 0.615, "id": 9}), ({"theta":[-0.2, 0.4], "h": [0.7,0.9], "id": 5}, {"p": 1, "id": 5}), ({"theta": [-1.1,0.9], "h": [0.75,-0.5], "id": 29}, {"p": -0.82, "id": 29}))
    @unpack
    def testEvalMixed(self, theta, expectedResult):
        result = targetCode.eval(theta)
        self.assertAlmostEqual(result["p"], expectedResult["p"], places = 5)
        self.assertEqual(result["id"], expectedResult["id"])
    
    #The test below is upon ready() function
    @data(({"p": 0.389998,"id": 1}, 10, 0.001, False), ({"p": 0.552004, "id":4}, 20, 0.001, False), ({"p": 1.13, "id": 5}, 31, 0.0001, False), ({"p": -1.083, "id": 7}, 1, 0.00001, False), ({"p": -1.085, "id": 7}, 34, 0.00001, True), ({"p": 0.8708800001, "id": 14}, 19, 0, True))
    @unpack
    def testready(self, p, t, threshold, expectedResult):
        ready = callableCode.readyClass(threshold)
        state = ready(p, t, self.P)
        self.assertEqual(state, expectedResult)

    #The three tests below are upon exploit() function with various dummy sample function
    @data(([1.3,1.5], {"theta": [0.6, -1.1], "h": [1.3,1.5], "id": 7}, {"p": -0.37, "id": 7}, [[1.3, 1.5], {"theta": [-0.23, -0.76], "h": [1.3, 1.5], "id": 7}]))
    @unpack
    def testexploit1(self, h, theta, p, expectedResult):
        exploit = callableCode.exploitClass(self.exploitSample1)
        hPrime, thetaPrime = exploit(h, theta, p, self.P)
        self.assertListEqual(hPrime, expectedResult[0])
        self.assertDictEqual(thetaPrime, expectedResult[1])
    
    @data(([0,1], {"theta": [0, 0.9], "h": [0,1], "id": 2}, {"p": 0.39, "id": 2}, [[0,1], {"theta": [0.9, 0], "h": [0,1], "id": 2}]))
    @unpack
    def testexploit2(self, h, theta, p, expectedResult):
        exploit = callableCode.exploitClass(self.exploitSample2)
        hPrime, thetaPrime = exploit(h, theta, p, self.P)
        self.assertListEqual(hPrime, expectedResult[0])
        self.assertDictEqual(thetaPrime, expectedResult[1])
    
    @data(([0.5,0.3], {"theta": [-0.9,-0.9], "h": [0.5,0.3], "id": 3}, {"p": -0.42, "id": 3}, [[0.5,0.3],{"theta": [-0.2, 0.4], "h": [0.5,0.3], "id":3}]))
    @unpack
    def testexploit3(self, h, theta, p, expectedResult):
        exploit = callableCode.exploitClass(self.exploitSample3)
        hPrime, thetaPrime = exploit(h, theta, p, self.P)
        self.assertListEqual(hPrime, expectedResult[0])
        self.assertDictEqual(thetaPrime, expectedResult[1])
    
    #The three tests below are upon explore() function with various dummy sample function
    @data(([0.7,0.5], {"theta": [0.13,0.24], "h": [0.7,0.5], "id": 11}, [[0.08,0.12], {"theta": [0.13,0.24], "h": [0.08,0.12], "id": 11}]))
    @unpack
    def testexplore1(self, hPrime, thetaPrime, expectedResult):
        explore = callableCode.exploreClass(self.exploreSample1)
        newh, newtheta = explore(hPrime, thetaPrime, self.P)
        self.assertAlmostEqual(newh[0], expectedResult[0][0])
        self.assertAlmostEqual(newh[1], expectedResult[0][1])
        self.assertAlmostEqual(newtheta["h"][0], expectedResult[1]["h"][0])
        self.assertAlmostEqual(newtheta["h"][1], expectedResult[1]["h"][1])

    @data(([2.4,0.35], {"theta": [-0.23, -0.76], "h": [2.4,0.35], "id": 10}, [[0,0], {"theta": [-0.23,-0.76], "h": [0,0], "id": 10}]))
    @unpack
    def testexplore2(self, hPrime, thetaPrime, expectedResult):
        explore = callableCode.exploreClass(self.exploreSample2)
        newh, newtheta = explore(hPrime, thetaPrime, self.P)
        self.assertListEqual(newh, expectedResult[0])
        self.assertDictEqual(newtheta, expectedResult[1])

    @data(([0.7,0.9], {"theta": [-0.2, 0.4], "h": [0.7,0.9], "id": 5}, [[-0.32, -0.3], {"theta": [-0.2, 0.4], "h": [-0.32,-0.3], "id": 5}]))
    @unpack
    def testexplore3(self, hPrime, thetaPrime, expectedResult):
        explore = callableCode.exploreClass(self.exploreSample3)
        newh, newtheta = explore(hPrime, thetaPrime, self.P)
        self.assertAlmostEqual(newh[0],expectedResult[0][0],places=5)
        self.assertAlmostEqual(newh[1],expectedResult[0][1],places=5)
        self.assertAlmostEqual(newtheta["theta"][0], expectedResult[1]["theta"][0])
        self.assertAlmostEqual(newtheta["theta"][1], expectedResult[1]["theta"][1])
        self.assertAlmostEqual(newtheta["h"][0], expectedResult[1]["h"][0])
        self.assertAlmostEqual(newtheta["h"][1], expectedResult[1]["h"][1])
        self.assertAlmostEqual(newtheta["id"], expectedResult[1]["id"])

    #The following test is upon update() function
    @data(({"theta" :[0.6, -1.1],"h": [1.2,1.1], "id": 7}, [1.2,1.1], {"p": -0.37,"id": 7}, 3), ({"theta": [0.28, -0.89], "h": [0.9,1.9], "id": 9}, [0.9,1.9], {"p": 0.3295, "id":9}, 15))
    @unpack
    def testupdate(self, theta, h, p, t):
        newP, newt = targetCode.update(self.P, theta, h, p, t)
        for counter, item in enumerate(newP):
            if item["p"]["id"] == p["id"]:
                self.assertDictEqual(item["theta"], theta)
                self.assertListEqual(item["h"], h)
                self.assertDictEqual(item["p"], p)
                self.assertEqual(item["t"], t+1)
        print(newP)
    
    #The following test is upon thetaHighestp() function 
    @data(({"theta":[0, 0], "h": [0.5,0.5], "id": 6}))
    def testthetaHighestp(self, expectedResult):
        maxTheta = targetCode.thetaHighestp(self.P)
        self.assertDictEqual(maxTheta, expectedResult)


if __name__ == '__main__':
    unittest.main(verbosity=2)


