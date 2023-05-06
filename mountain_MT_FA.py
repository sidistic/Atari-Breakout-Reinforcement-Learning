from Tilecoder import numTilings, tilecode, numTiles
from Tilecoder import numTiles as n
from pylab import *  #includes numpy
import random
import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

numRuns = 1
numEpisodes = 2000
alpha = 0.05/numTilings
gamma = 1
lmbda = 0.9
Epi = Emu = epsilon = 0.1
n = numTiles * 3 # 4 * 9 * 9 * 3
F = [-1]*numTilings # 4

# for average purpose
aveReturn = zeros((numRuns,numEpisodes))
aveStep = zeros((numRuns,numEpisodes))

def egreedy(Qs,epsilon):
    if rand() < epsilon: return randint(3) # return a random action
    else:                return argmax(Qs) # return argmax θt^T ф(S,A)
    
# def expQunderPi(Qs,method):
#     if method=="sarsa":
#         if rand() < epsilon: return Qs[randint(3)] # return a random action
#         else:                return max(Qs)

#     return max(Qs)

def Qs(F):
    # initialize Q[S, A, F]
    # S = (position, velocity); F = ф
    Q = np.zeros(3)
    
    # numActions
    # for every possible action a in F
    for a in range(3):
        # numTilings
        for i in F:
            # update Qa
            Q[a] = Q[a] + theta[i + (a * numTiles)]
    return Q
            
runSum = 0.0


method = input("0 for Sarsa,1 for Q-Learning :")
#Multiple tile coding with offset
for run in range(numRuns):
    theta = -0.01*rand(n)
    returnSum = 0.0
    
    for episodeNum in range(numEpisodes):
        G = 0

        step = 0

        # initialize state
        S = env.reset()

        # get a list of four tile indices
        tilecode(S[0], S[1], F)
        
        Q = Qs(F)

        # pick the action
        A = egreedy(Q, Emu)
        
        # initialize e (eligibility trace vector)
        e = np.zeros(n)
        
        done = False
        # repeat for each step of episode
        while done is not True:       
            

            # get a list of four tile indices
            tilecode(S[0], S[1], F)
            
            Q = Qs(F)

            # observe reward, and next state
            Sprime, R, done ,_ = env.step(A)
            
            delta = R - Q[A]
            
            G = G + R
            
            for i in F:
                # replacing traces
                e[i + (A*numTiles)] = 1
            
            # if S' is terminal, then update theta; go to next episode
            if done == True:
                theta = theta + alpha * delta * e
                break
            
            tilecode(Sprime[0], Sprime[1], F)
            
            Qprime = Qs(F)
            
            if(method == "0"):
                new_A = egreedy(Qprime,Emu)
                delta += gamma * Qprime[new_A]
            else:
                delta += gamma * max(Qprime)
                new_A = egreedy(Qprime,Emu)

            # update theta
            theta = theta + alpha * delta * e
            
            # # update e
            e = np.zeros(n)
            # e = gamma * lmbda * e
            
            # update current state to next state for next iteration
            S = Sprime

            A = new_A
            
            step = step + 1
        
        print("Episode: ", episodeNum, "Steps:", step, "Return: ", G)

        # average 
        aveReturn[run][episodeNum] = G
        aveStep[run][episodeNum] = step

        returnSum = returnSum + G
    print("Average return:", returnSum/numEpisodes)
    runSum += returnSum

print("Overall performance: Average sum of return per run:", runSum/numRuns)

def writeAve(filename, array):
    fout = open(filename, 'w')
    # ave = zeros(numEpisodes)
    for i in range(numEpisodes):
        fout.write(repr(array[0][i]) + '\n')
        # fout.write('\n')
    fout.close()

def writeF():
    fout = open('value', 'w')
    F = [0]*numTilings
    steps = 50
    for i in range(steps):
        for j in range(steps):
            tilecode(-1.2+i*1.7/steps, -0.07+j*0.14/steps, F)
            height = -max(Qs(F))
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()

writeF()
writeAve("Return_MT_FA", aveReturn)
writeAve("Step_MT_FA", aveStep)