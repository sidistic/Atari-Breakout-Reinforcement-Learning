from Breakout_tilecoder import numTilings, tilecode, numTiles
from pylab import *  #includes numpy
import numpy as np
import gym
import matplotlib.pyplot as plt

#Initializations
env = gym.make('Breakout-v0')
env.reset()
num_actions = 4

numRuns = 1
numEpisodes = 2500
alpha = 0.0001
print(alpha)
gamma = 1
lmbda = 0.9
Epi = Emu = epsilon = 0.4
reduction = Epi/(numEpisodes*100.0)
n = numTiles * num_actions
F = [-1]*numTilings # 4
# Actions:
# 0: do nothing
# 1: start game
# 2: move right
# 3: move left

# for average purpose
aveReturn = zeros((numRuns,numEpisodes))
aveStep = zeros((numRuns,numEpisodes))

def egreedy(Qs,epsilon):
    global num_actions
    if rand() < epsilon: return randint(num_actions) # return a random action
    else:                return argmax(Qs) # return argmax θt^T ф(S,A)
# after downsizing and greyscale trans the image, the size is 105x80
# the score part ends after the first 9 rows, Then the boundary shows
# for the next 7 rows. the size of the image after preprocessing and
# croping is 82x72

# color of player and ball in 114 i.e red in greyscale

# extract_state checks the image extracts the state
# which consists of the x,y position of the ball and
# x position of the player
def extract_state(I):
    I = preprocess(I)
    found_ball = False
    ball_pos = [0,0]
    for i in range(0,len(I) - 4): # the last 3 rows have the block
        # 13,14,15 row is a red tile row we skip this row as its the same color as the ball
        for j in range(0,len(I[i])):
            if I[i][j] == 114 and i == 14 :
                continue
            if I[i][j] == 114 :
                if I[i+1][j] == 114:
                    found_ball = True
                    ball_pos = [i,j]
                    if I[i+2][j] == 114:
                        found_ball = False
                        ball_pos = [0,0]
                        if I[i+3][j] == 114 and I[i+4][j] == 114:
                            found_ball = True
                            ball_pos = [i,j]
    player_pos = 0
    for i in range(len(I)-3,len(I)): # find the position of the player in the last 3 rows
        for j in range(0,len(I[i])-2): # 8 is the length of the player
            # print(I[i][j:j+8],end=" ")
            if(np.sum(I[i][j:j+2]) == 114*2):
                player_pos = j+2
                break
        # print()
    # print(ball_pos,player_pos)
    return [ball_pos[0]*72+ball_pos[1],player_pos]




def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

# preprocess reduces the size, crops the borders and the score and gives just the playable part
def preprocess(img):
    img = to_grayscale(downsample(img))
    img = img[9+7:(len(img)-7)] # 9 is the score part which we crop and 7 border size in top and bottom
    for i in range(4): # border size is 4
        img = np.delete(img,0,1)
        img = np.delete(img,len(img[0])-1,1)
    return img

def Qs(F):
    # initialize Q[S, A, F]
    # S = (position, velocity); F = ф
    Q = np.zeros(num_actions)
    
    # numActions
    # for every possible action a in F
    for a in range(num_actions):
        # numTilings
        for i in F:
            # update Qa
            Q[a] = Q[a] + theta[i + (a * numTiles)]
    return Q
            
runSum = 0.0

# observation, reward, done, info = env.step(0)
# for i in range(100):
#     env.render()
#     observation, reward, done, info = env.step(2)
#     print(extract_state(observation), done)
# observation = preprocess(observation)
# for i in observation:
#     for j in i:
#         print(j,end = " ")
#     print()
# print (observation)

method = input("0 for Sarsa,1 for Q-Learning :")
#Multiple tile coding with offset
for run in range(numRuns):
    theta = -0.01*rand(n)
    print(len(theta),theta)
    returnSum = 0.0
    
    for episodeNum in range(numEpisodes):
        G = 0

        step = 0

        # initialize state
        S = env.reset()

        S = extract_state(S)

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
            env.render()
            # get a list of four tile indices
            tilecode(S[0], S[1], F)
            
            Q = Qs(F)

            # observe reward, and next state
            Sprime, R, done ,_ = env.step(A)
            Sprime = extract_state(Sprime)
            
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

        # Epi = Emu = epsilon = Epi - reduction
        
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
            tilecode(0+i*41*6/steps, 0+j*36/steps, F)
            height = -max(Qs(F))
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()

writeF()
writeAve("breakout_aveReturn", aveReturn)
writeAve("breakout_aveStep", aveStep)