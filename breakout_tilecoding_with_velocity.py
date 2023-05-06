from breakout_tilecoder_with_velocity import numTilings, tilecode, numTiles
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
print(n)
F = [-1]*numTilings # 4

# for average purpose
aveReturn = zeros((numRuns,numEpisodes))
aveStep = zeros((numRuns,numEpisodes))

l4_states = [[[0,0],0],[[0,0],0],[[0,0],0],[[0,0],0]]

def get_velocity(last_states):
    frame_by_frame_vel_x = []
    frame_by_frame_vel_y = []
    for i in range(1,len(last_states)):
        xvel = last_states[i][0][0] - last_states[i-1][0][0]
        yvel = last_states[i][0][1] - last_states[i-1][0][1]
        if xvel > 5 or xvel < -5 : xvel = 0
        if yvel > 5 or yvel < -5 : yvel = 0
        frame_by_frame_vel_x.append(float(xvel))
        frame_by_frame_vel_y.append(float(yvel))
    return [np.average(frame_by_frame_vel_x),np.average(frame_by_frame_vel_y)]

def update_last_states(last_states,new_state):
    last_states = last_states[1:]
    last_states.append(new_state)
    # print(last_states)
    return last_states

        

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
    return [ball_pos,player_pos]




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

# observation, reward, done, info = env.step(0)
# for i in range(100):
#     env.render()
#     observation, reward, done, info = env.step(1)
#     new_state = extract_state(observation)
#     l4_states = update_last_states(l4_states,new_state)
#     # print(l4_states)
#     vel = get_velocity(l4_states)
#     print(new_state,vel)


def egreedy(Qs,epsilon):
    global num_actions
    if rand() < epsilon: return randint(num_actions) # return a random action
    else:                return argmax(Qs) # return argmax θt^T ф(S,A)

def Qs(F):
    # initialize Q[S, A, F]
    # S = (position, velocity); F = ф
    # print(F)
    Q = np.zeros(num_actions)
    # print(F)
    # numActions
    # for every possible action a in F
    for a in range(num_actions):
        # numTilings
        # print(a)
        for i in F:
            # update Qa
            k = (i + (a * numTiles))
            # print(i,a,numTiles)
            if(k>=n):
                k=n-1
            # print(k)
            Q[a] = Q[a] + theta[k]
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
        S = extract_state(S)
        l4_states = update_last_states(l4_states,S)
        Sv = get_velocity(l4_states)

        # get a list of four tile indices
        tilecode(S[0][0], S[0][1],S[1],Sv[0],Sv[1], F)
        
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
            tilecode(S[0][0], S[0][1],S[1],Sv[0],Sv[1], F)
            
            Q = Qs(F)

            # observe reward, and next state
            Sprime, R, done ,_ = env.step(A)
            Sprime = extract_state(Sprime)
            l4_states = update_last_states(l4_states,Sprime)
            Sprimev = get_velocity(l4_states)
            
            delta = R - Q[A]
            
            G = G + R
            
            for i in F:
                # replacing traces
                k = i + (A*numTiles)
                # print(i,A,numTiles)
                if(k>=n):
                    k=n-1
                e[k] = 1
            
            # if S' is terminal, then update theta; go to next episode
            if done == True:
                theta = theta + alpha * delta * e
                break
            
            tilecode(Sprime[0][0], Sprime[0][1],Sprime[1],Sprimev[0],Sprimev[1], F)
            
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
            Sv = Sprimev

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

# def writeF():
#     fout = open('value_vel', 'w')
#     F = [0]*numTilings
#     steps = 50
#     for i in range(steps):
#         for j in range(steps):
#             tilecode(0+i*41*6/steps, 0+j*36/steps, F)
#             height = -max(Qs(F))
#             fout.write(repr(height) + ' ')
#         fout.write('\n')
#     fout.close()

# writeF()
writeAve("Breakout_aveReturn_vel", aveReturn)
writeAve("Breakout_aveStep_vel", aveStep)