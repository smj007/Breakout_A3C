# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

# Implementing a function to make sure the models share the same gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad 


def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)   #shifting the seed with rank to asynchronize each training agent
    env = create_atari_env(params.env_name) #breakout-V0
    env.seed(params.seed + rank) #aligning the seed of the environment on the seed of the agent
    model = ActorCritic(env.observation_space.shape[0], env.action_space) 
    state = env.reset() #state is the inp image, which is a np array of 1*42*42 (grayscale)
    state = torch.from_numpy(state)
    done = True #to indicate when the game is done
    
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict()) #synchronizing with the shared model - the agent gets the shared model to do an exploration on num_steps
        if done:
            #reset the LSTM cell and hidden states to 0
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
            
        else:
            
            cx = Variable(cx.data)
            hx = Variable(hx.data)
            
        #values to be calc in each step as exploration goes on
        values = [] #V(s)
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(params.num_steps):
            #going through the num_steps exploration steps
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            entropy = -(log.prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial.data() #selecting a random draw from the distribution obtained
            log_prob = log_prob.gather(1, Variable(action)) #get the log_prob correspoding to the action chosen
            log_probs.append(log_prob)
            values.append(value)
            state, reward, done, _ = env.step(action.numpy())
            done = (done or episode_length >= params.max_episode_length) #done if the agent gets stuck for too long
            reward = max(min(reward, 1), -1) #clip between -1 and 1
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break #we stop the exploration and we directly move on to the next step: the update of the shared model
        R = torch.zeros(1, 1) #cumulative reward init
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data #value of the last reached state s is passed onto the cumReward  (.data is used as we get the data part from unsqueezed tensor)
        values.append(Variable(R)) #storing the value of the last reached state s
        policy_loss = 0
        value_loss = 0
        R = Variable(R) #asserting that its a torch var
        gae = torch.zeros(1, 1) #this is the generalized advantage estimation, which is A(a, s) = Q(s, a) - V(s)
        
        #now we start from the lasyt exploration and go back in time to update
        for i in reversed(range(len(rewards))):
            #we explore based on rewards only
            R = params.gamma * R + rewards[i]  # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
                #here already in the first loop , R = V(last state)
            advantage = R - values[i]  # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]  
            value_loss = value_loss + 0.5 * advantage.pow(2)
            #we need policy loss, so we need gae, for which we need Temporal diff TD
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD #gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
            
        optimizer.zero_grad()
        loss = (policy_loss + 0.5 * value_loss)
        loss.backward() #as policy loss is smaller and we are trying to minimise, give 2x importance to that than value loss
        torch.nn.utils.clip_grad_norm(model.parameters(), 40) #clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
        ensure_shared_grads(model, shared_model) #ensure that the agents' model and the shared model are using same grads
        optimizer.step() 