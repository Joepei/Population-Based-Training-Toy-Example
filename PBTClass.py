import time

class PBTTrain:
    def __init__(self, step, eval, ready, exploit, explore, update, thetaHighestp, endofTrain, Population):
        self.step = step
        self.eval = eval
        self.ready = ready
        self.exploit = exploit
        self.explore = explore
        self.update = update
        self.thetaHighestp = thetaHighestp
        self.P = Population
        self.endofTrain = endofTrain
        
    def __call__(self, worker):
        initial, QList, thetaList, bestthetaList = worker
        theta, h, p, t  = (initial["theta"], initial["h"], initial["p"], initial["t"])
        while not self.endofTrain(t): 
                thetaList.put(theta["theta"][:])
                theta = self.step(theta, h)
                p = self.eval(theta)
                if self.ready(p,t,self.P):
                    QList.put(1.2-theta["theta"][0]**2-theta["theta"][1]**2)
                    hPrime, thetaPrime = self.exploit(h, theta, p, self.P)
                    if theta != thetaPrime:
                        h, theta = self.explore(hPrime, thetaPrime, self.P)
                        p = self.eval(theta)
                self.P, t = self.update(self.P, theta, h, p, t)
        time.sleep(50)
        bestthetaList.put(self.thetaHighestp(self.P))
        return (self.thetaHighestp(self.P))
    








    




