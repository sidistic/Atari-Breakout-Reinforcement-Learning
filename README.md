# Atari-Breakout-Reinforcement-Learning
# Algorithms Used in Atari Breakout Agent

In the Atari Breakout agent implementation, the following algorithms were utilized to solve the problem:

## Multiple Tile Coding

The Multiple Tile Coding algorithm is a function approximation technique that divides the state space into multiple overlapping tiles. Each tile represents a binary feature and is associated with a weight. During training, the agent updates the weights based on the observed rewards and tries to learn the optimal policy. In the case of the Atari Breakout agent, Multiple Tile Coding was employed to capture relevant features from the processed image, such as the positions of the ball and the player. By representing the state using multiple tiles, the algorithm aimed to learn the optimal actions to maximize the score.

## Radial Basis Functions

Radial Basis Functions (RBF) is another function approximation method used in reinforcement learning. It involves placing Gaussian-shaped basis functions throughout the state space. Each basis function represents a particular feature, and its activation depends on the distance between the current state and the center of the function. In the context of the Atari Breakout agent, RBF was utilized to approximate the Q-values of different states. By placing basis functions strategically and updating their weights during training, the algorithm aimed to learn an accurate value function and make optimal decisions in the game.

## REINFORCE Algorithm

The REINFORCE algorithm, also known as the Monte Carlo Policy Gradient, is a policy-based method for reinforcement learning. It directly learns the policy of the agent without estimating a value function. In this algorithm, the agent interacts with the environment, collects trajectories, and computes the gradients of the policy based on the observed rewards. The gradients are then used to update the policy parameters, gradually improving the policy over time. For the Atari Breakout agent, the REINFORCE algorithm was employed to learn a policy that maximizes the score in the game. However, it is mentioned in the report that the REINFORCE algorithm did not show significant learning progress even after trying various parameter combinations.

These algorithms were applied to the Atari Breakout game in order to develop an agent capable of maximizing its score. Each algorithm had its own approach to function approximation and policy optimization, aiming to learn and improve the agent's performance over time. The report provides insights into the performance and effectiveness of each algorithm in solving the Breakout game.

# Breakout Atari

The objective in the Breakout Atari game within the OpenAI Gym environment is to maximize the player's score. The observation provided by the environment is an RGB image of the game screen, represented as a 3D array with dimensions (210, 160, 3). Each action is repeated for a duration of kk frames, where kk is uniformly sampled from the set {2, 3, 4}.

To process the image and extract relevant information, such as the ball position and the player position, the following steps were performed:

1. Downsizing the image by a factor of 2.
2. Conversion to grayscale.
3. Cropping the image to remove the score and borders.
4. Reading the processed image pixel by pixel to identify the positions of the player and the ball.
   - The ball and player are identified by checking if the pixel value is 114.
   - A red block line with the same color value is present in the image, but the code was designed to find the ball even if it had the same color.

Additionally, the velocity of the ball was calculated using four consecutive images. The average displacement of the ball's position in the x and y dimensions over the frames was used as the velocity.

### Parameters and Execution

For all algorithms, the code was run for 2500 episodes. The following parameters were used where applicable:

- Alpha: 0.0001
- Gamma: 1
- Lambda: 0.9
- Epi = Emu = epsilon: 0.4

## Algorithm Comparison

The algorithms were evaluated using two variations: one with velocity and one without velocity. The results are as follows:

### TileCoding with Velocity

- Average Return: 0.9524
- Average Step: 236.3116

### TileCoding without Velocity

- Average Return: 1.316
- Average Step: 265.386

### REINFORCE

- Average Return: 1.4596
- Average Step: 256.6224

Looking at the averages, it is evident that including velocity in the TileCoding algorithm did not significantly improve performance. However, it should be noted that incorporating velocity increased the state dimensionality, requiring longer runtimes to achieve meaningful learning.

The REINFORCE algorithm performed the best and demonstrated learning progress. Graphs were generated to visualize the results, although they might not provide clear insights.

When the 0 reward case was replaced with a penalty of -10, the algorithm showed an increasing frequency of 4 rewards as the number of episodes increased. This indicates that the algorithm was indeed learning and could potentially be further improved by running it for a larger number of episodes (possibly millions) and fine-tuning the parameters.

Please refer to the project code and associated graphs for more detailed information.
