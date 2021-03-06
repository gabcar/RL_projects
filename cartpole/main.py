from collections import namedtuple
import random 
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import numpy as np

from network import Mlp


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(torch.tensor(state, dtype=torch.float32, device=device)))
    else:
        return torch.tensor(random.randrange(2), device=device, dtype=torch.long)

def main():
    # init the environment
    # gym.envs.register(
    # id='CartPole-v11',
    # entry_point='gym.envs.classic_control:CartPoleEnv',
    # tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
    # reward_threshold=475.0,
    # )
    env = gym.make('MountainCar-v0')

    # env = gym.make('CartPole-v0')

    obs = env.reset()
    # placeholders
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    buffer = ReplayMemory(10000)
    policy_net = Mlp(state_dim, [256], action_dim).to(device)
    target_net = Mlp(state_dim, [256], action_dim).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    # Loss and Optimizer
    # Note that the optimizer keeps track of the 
    # graph of the network (mlp.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    n_episodes = 30000
    avg_steps_2 = []
    for i_episode in range(n_episodes):
        obs = env.reset()
        steps = 0
        reward = 0
        for i in range(200):
            if np.mean(avg_steps_2) < 200:
                env.render()

            action = select_action(obs, policy_net).numpy()

            obs_, re_, done, _ = env.step(action)
            #reward += re_
            if done and i < 199:
                reward = 200 - i
                print(reward)
            else: 
                reward = 0
            # reward = -np.abs(obs_[2])

            buffer.push(torch.tensor(obs, dtype=torch.float32, device=device), 
                        torch.tensor((int(action),), dtype=torch.long, device=device),
                        torch.tensor((reward,), dtype=torch.float32, device=device),
                        torch.tensor(obs_, dtype=torch.float32, device=device))

            if len(buffer) >= BATCH_SIZE:
                transitions = buffer.sample(BATCH_SIZE)
                state_batch, action_batch, reward_batch, next_state_batch = zip(*transitions)

                # to get the output of the network we simply pass the batch 
                # through the net with a function call
                state_batch = torch.cat(state_batch).view(-1, state_dim)
                action_batch =  torch.cat(action_batch).view(-1, 1)
                reward_batch =  torch.cat(reward_batch).view(-1, 1)
                next_state_batch =  torch.cat(next_state_batch).view(-1, state_dim)
                
                current_q_values = policy_net(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    max_next_q_values = target_net(next_state_batch).max(1)[0]
                expected_q_values = reward_batch + (GAMMA * max_next_q_values)

                loss = F.smooth_l1_loss(current_q_values, expected_q_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            obs = obs_
            steps += 1
            if done:
                avg_steps_2.append(steps)
                avg_steps_2 = avg_steps_2[-50:]
                print("{} - steps: {}, - avg. steps: {}".format(i_episode, steps, np.round(np.mean(avg_steps_2))))
                break
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


if __name__ == '__main__':
    main()