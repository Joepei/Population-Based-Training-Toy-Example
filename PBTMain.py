import sys
sys.path.append('../ToyPBT')
import numpy as np
import PBTToyExample as exampleCode
import PBTClass as algoCode
import random
import pylab as plt
import multiprocessing
import pickle
from PBTCallable import stepClass
from PBTCallable import sampleExploreClass
from PBTCallable import readyClass
from PBTCallable import exploitClass
from PBTCallable import exploreClass
from PBTCallable import endofTrainClass
from PBTCallable import sampleExploitClass
from queue import Queue
import time

def plotQ(QLists):
    Q0 = QLists[0]
    Q1 = QLists[1]
    index = list(range(0,len(Q0)))
    fig = plt.figure()
    plt.plot(index, Q0)
    plt.plot(index, Q1, 'r-')
    plt.ylim(0, 1.2)
    plt.xlabel("steps")
    plt.ylabel("Q(theta)")
    plt.title("PBT")
    plt.xticks(range(0,len(Q0)))
    plt.show()
    fig.savefig("PBT_Q.png")

def plottheta(thetaLists):
    theta0 = thetaLists[0]
    theta1 = thetaLists[1]
    theta0 = np.array(theta0)
    theta1 = np.array(theta1)
    fig = plt.figure()
    plt.plot(theta0[:, 0], theta0[:, 1], "o", color = "black")
    plt.plot(theta1[:, 0], theta1[:, 1], "o", color = "red")
    x_vals = np.linspace(0, 1, 1000)
    y_vals = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.sqrt(X**2+Y**2)
    cp = plt.contourf(X, Y, Z, levels = 20, cmap = "Oranges")
    plt.xlabel("Theta0")
    plt.ylabel("Theta1")
    plt.title("PBT")
    plt.show()
    fig.savefig("PBT_theta.png")

def dump_queue(queue):
    result = []
    while not queue.empty():
        item = queue.get()
        result.append(item)
    return(result)

def process(trainer, Population):
        pop_length = len(Population)
        """
        QQueues = [] #create Queues here
        #QQueues  = multiprocessing.Queue()
        thetaQueues = []
        ps = []
        workers = []
        """
        bestthetaQueue = multiprocessing.Queue()
        #create a function
        """
        for i in range(pop_length): #Use enumerate
            QQueues.append(multiprocessing.Queue())
            thetaQueues.append(multiprocessing.Queue())
            initial = Population[i]
            workers.append((initial, QQueues[i], thetaQueues[i], bestthetaQueue))
            ps.append(multiprocessing.Process(target = trainer, args = (workers[i],)))
        """
        def prepare(initial, bestthetaQueue):
            QQueue = multiprocessing.Queue()
            thetaQueue = multiprocessing.Queue()
            worker = (initial, QQueue, thetaQueue, bestthetaQueue)
            process = multiprocessing.Process(target = trainer, args = (worker,))
            return(QQueue, thetaQueue, process)
        
        QueueList = [prepare(initial, bestthetaQueue) for initial in Population]
        QQueues = [QQueue for (QQueue, thetaQueue, process) in QueueList]
        thetaQueues = [thetaQueue for (QQueue, thetaQueue, process) in QueueList]
        ps = [process for (QQueue, thetaQueue, process) in QueueList]
        
        """
        for initial in Population:
            QQueue = multiprocessing.Queue()
            thetaQueue = multiprocessing.Queue()
            worker = (initial, QQueue, thetaQueue, bestthetaQueue)
            process = multiprocessing.Process(target = trainer, args = (worker,))
            QQueues.append(QQueue)
            thetaQueues.append(thetaQueue)
            workers.append(worker)
            ps.append(process)
        """
            
        startTime = time.time()
        for process in ps:
            process.start()
            time.sleep(0.015)
        
        for process in ps:
            process.join()
        endTime = time.time()
        print(endTime - startTime)
        """
        #get all the lists here
        for i in range(pop_length):
            ps[i].start()
            time.sleep(0.015)
        for i in range(pop_length):
            ps[i].join()
        """

        QLists = []
        thetaLists = []
        for i in range(pop_length):
            QLists.append(dump_queue(QQueues[i]))
            thetaLists.append(dump_queue(thetaQueues[i]))
        bestthetaList = dump_queue(bestthetaQueue)

        plotQ(QLists)
        plottheta(thetaLists)
        return(bestthetaList)


def main():
    alpha = 0.05 #learning rate which can self-define
    threshold = 0.0001 #improvement threshold which can self-define
    numIt = 200 #Criteria to determine convergence
    step = stepClass(alpha)
    eval = exampleCode.eval
    ready = readyClass(threshold)
    sampleExploit = sampleExploitClass() #np.random.sample(P, n)[0]
    exploit = exploitClass(sampleExploit)
    sampleExplore = sampleExploreClass() #np.random.uniform(0.8, 1.2, n)
    explore = exploreClass(sampleExplore)
    update = exampleCode.update
    thetaHighestp = exampleCode.thetaHighestp
    endofTrain = endofTrainClass(numIt)
    
    with multiprocessing.Manager() as manager:
        Population = manager.list([manager.dict({"theta": manager.dict({"theta": manager.list([0.9, 0.9]),"h": manager.list([1, 0]), "id": 1}), \
                                                "h": manager.list([1, 0]), "p": manager.dict({"p": -0.42, "id": 1}), "t": 0 }), \
                                   manager.dict({"theta": manager.dict({"theta": manager.list([0.9, 0.9]),"h": manager.list([0,1]),"id": 2}), \
                                                "h": manager.list([0,1]), "p": manager.dict({"p": -0.42,"id": 2}), "t": 0})]) #Initialization
        trainer = algoCode.PBTTrain(step, eval, ready, exploit, explore, update, thetaHighestp, endofTrain, Population)
        bestthetaList = process(trainer, Population)


if __name__ == '__main__':
    main()

