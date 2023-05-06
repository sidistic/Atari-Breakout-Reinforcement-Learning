import matplotlib.pyplot as plt
import numpy as np

fil = open("Breakout_step_RL")
ret_RL = []
for line in fil:
    x = float(line)
    if(x==-10):
        x=0
    ret_RL.append(x)
fil.close()

# fil = open("breakout_aveReturn")
# ret_MT = []
# for line in fil:
#     ret_MT.append(float(line))
# fil.close()

# fil = open("Step_RBF_FA")
# ret_RBF = []
# for line in fil:
#     ret_RBF.append(float(line))
# fil.close()
print(np.average(ret_RL))
# plt.plot(ret_MT,label="TileCoding")
plt.plot(ret_RL,label="TileCoding with Velocity")
# plt.plot(ret_RL,label="REINFORCE")
plt.show()