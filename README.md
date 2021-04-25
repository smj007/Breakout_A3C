<div align = "center">
    
### ☢️ Breakout - A3C + LSTM :radioactive:
</div>


**Breakout - A3C + LSTM**

An A3C implementation to ace the Breakout game by Atari, modified with the addition of an LSTM for improved performance

**To run :**
- Download the ```Breakout_Run``` colab notebook 
- Upload the .py scripts onto drive and mount through colab
- Follow the code cells in the notebook, modifying the file location accordingly

   _Breakout_ is an arcade game developed and published by Atari, Inc., and released on May 13, 1976. Our goal in this project is to use Reinforcement learning to train AI Agent to play Breakout game.

![](https://miro.medium.com/max/1392/1*YtnGhtSAMnnHSL8PvS7t_w.png)

**Algorithm:** Asynchronous Advantage Actor-Critic (A3C)

**Asynchronous:** Rather than in a [DQN](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df), where there is a single instance of an agent, and the environment, A3C utilizes several copies of the above in order to have a larger experience. In A3C there is a global network, and several agents which each have their own set of network parameters. Each of these agents interacts with their own copy of the environment at the same time as the other agents are interacting with their environments, learning individually. The reason this works better than having a single agent, is that the experience of each agent is independent of the experience of the others. In this way the overall experience available for training becomes more diverse, and it gets easier to tackle the game.

**Actor-Critic:** In A3C, we will be estimating both a value function V(s) (how good a certain state is to be in) and a policy π(s) (a set of action probability outputs). Each of these are separate fully-connected layers at the top of the network. Critically, the agent uses the value estimate (the critic) to update the policy (the actor) more intelligently than traditional policy gradient methods.

**Advantage:** The update rule uses the discounted returns from a set of experiences in order to tell the agent which of its actions were good, and which were not. The network was then updated in order to encourage and discourage actions appropriately.

```Discounted Reward: R = γ(r)```

The notion of using advantage in the algorithm, rather than just discounted returns is to allow the agent to determine not just how good its actions were, but how much better they turned out to be than the pre-set benchmark. This allows the algorithm to focus on where the network’s predictions were lacking. In the case of [Dueling Q-Network architecture](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df), the advantage function is as follows:

```Advantage: A = Q(s,a) - V(s)```

Q values are not being found here, so we can use the discounted returns (R) as an estimate of Q(s,a) to allow us to generate an estimate of the advantage.

```Advantage Estimate: A = R - V(s)```



**References:**

[1] [https://arxiv.org/pdf/1506.02438.pdf](https://arxiv.org/pdf/1506.02438.pdf)

[2] [https://arxiv.org/pdf/1607.05077](https://arxiv.org/pdf/1607.05077)


**Future Enhancements:**
   - UNREAL Algorithm (to beat A3C + LSTM) 
