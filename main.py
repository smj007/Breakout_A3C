# Main code

from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
from envs import create_atari_env
from model import ActorCritic
from train import train
from test import test
import my_optim

# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_steps = 20
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0'

# Main run
os.environ['OMP_NUM_THREADS'] = '1'  #1 thread will run per core
params = Params() 
torch.manual_seed(params.seed) #set seed 
env = create_atari_env(params.env_name) #optimized env , due to universe package
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
shared_model.share_memory() #model is stored in the computer shared memory, so that any process in any thread (agent), even if they are in diff cores, can access the model
optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr) #the optimizer is also shared because it acts on the shared model
optimizer.share_memory() #store the optimizer in the shared memory so that all the agents can have access to this shared memory to optimize the model
processes = []
p = mp.Process(target=test, args=(params.num_processes, params, shared_model)) #to run the test process
p.start()
processes.append(p)
for rank in range(0, params.num_processes): #loop to run all the train processes
    p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
