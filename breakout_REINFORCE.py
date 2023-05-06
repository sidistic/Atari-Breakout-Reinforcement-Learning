from pylab import *  #includes numpy
import numpy as np
import gym
import matplotlib.pyplot as plt

#Hyperparameters
NUM_EPISODES = 2500
LEARNING_RATE = 0.0001
GAMMA = 0.99

#Initializations
env = gym.make('Breakout-v0')
env.reset()
nA = 4
dim = 2 #ball position x*width + ball pos y and player position
# Init weight
w = np.random.rand(dim, nA)

# Keep stats for final print of graph
episode_rewards = []

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
    return [(ball_pos[0]*72+ball_pos[1])/(82*72),(player_pos)/72]

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

# Our policy that maps state to action parameterized by w
def policy(state,w):
    z = state.dot(w)
    # print("z",z) 
    exp = np.exp(z/2)
    # print("ha",exp/np.sum(exp))
    return exp/np.sum(exp)

# Vectorized softmax Jacobian
def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

fout = open("Breakout_ret_RL",'w')
foutstep =open("Breakout_step_RL","w")
for e in range(NUM_EPISODES):

    state = env.reset()
    state = np.asarray(extract_state(state))[None,:]
    grads = []	
    rewards = []
    # Keep track of game score to print
    score = 0
    st =0
    while True:

		# Uncomment to see your model train in real time (slower)
		# env.render()
        env.render()

		# Sample from policy and take action in environment
        probs = policy(state,w)
        # print(state)
        # print(probs)
        action = np.random.choice(nA,p=probs[0])
        next_state,reward,done,_ = env.step(action)
        st+=1
        if(score==0 and done == True):
            reward=-10
        # print(extract_state(next_state),end=" ")
        next_state = np.asarray(extract_state(next_state))[None,:]

        # Compute gradient and save with reward in memory for our weight updates
        dsoftmax = softmax_grad(probs)[action,:]
        dlog = dsoftmax / probs[0,action]
        grad = state.T.dot(dlog[None,:])

        grads.append(grad)
        rewards.append(reward)		

        score+=reward

        # Dont forget to update your old state to the new state
        state = next_state

        if done or score < -1000:
            break

	# Weight update
    for i in range(len(grads)):

		# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
        w += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** r) for t,r in enumerate(rewards[i:])])
	# Append for logging and print
    episode_rewards.append(score)
    fout.write(str(score))
    fout.write('\n')
    foutstep.write(str(st))
    foutstep.write('\n')
    print("EP: " + str(e) + " Score: " + str(score) + "         ",probs[0]) 

fout.close()
foutstep.close()
plt.plot(np.arange(NUM_EPISODES),episode_rewards)
plt.title("MountainCar a=0.000025")
plt.show()
env.close()
