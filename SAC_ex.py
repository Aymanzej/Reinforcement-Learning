# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:43:09 2025

@author: julien.hautot
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import math
import os

# -------------------------
# Hyperparamètres (à adapter pour TP)
# -------------------------
ENV_NAME = "Hopper-v5"
SEED = 42
MAX_STEPS = ??      
START_STEPS = ??        
UPDATE_AFTER = ??       
UPDATE_EVERY = ??         
BATCH_SIZE = ??
GAMMA = ??
TAU = ??               
POLICY_LR = ??
Q_LR = ??
ALPHA_LR = ??
HIDDEN = ??
REPLAY_SIZE = ??
AUTOMATIC_ENTROPY_TUNING = True   

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# -------------------------
# Utils : replay buffer
# -------------------------
Transition = namedtuple("Transition", ("s", "a", "r", "s2", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Networks
# -------------------------
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=HIDDEN, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            ??
        )
        self.net.apply(weight_init)

    def forward(self, x):
        ??

# Critic (Q network) : prend state et action
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = ??
    def forward(self, s, a):
        ??

# Policy : retourne action sampleable et log_prob (avec correction tanh)
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = MLP(??)  
        self.action_dim = action_dim

    def forward(self, s):
        x = self.shared(s)
        mu, log_std = ??
        log_std = torch.tanh(??)
        # scale log_std to sensible range
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = ??
        return mu, std

    def sample(self, s):
        mu, std = self.forward(s)
        dist = ??
        z = ??                    
        action = ??                 
        
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)  



# -------------------------
# Agent SAC
# -------------------------
class SACAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.act_limit = float(env.action_space.high[0])

        # networks
        self.policy = ??
        self.q1 = ??
        self.q2 = ??
        self.q1_target = ??
        self.q2_target = ??

        # copy params to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=POLICY_LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=Q_LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=Q_LR)

        # automatic entropy tuning
        if AUTOMATIC_ENTROPY_TUNING:
            # target_entropy = -|A|
            self.target_entropy = ??
            # log alpha as parameter
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=DEVICE)
            self.log_alpha = torch.nn.Parameter(self.log_alpha)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        else:
            self.alpha = 0.2

    def select_action(self, state, evaluate=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if evaluate:
                _, _, mu = ??
                action = ??
                logp = None
            else:
                a, logp, _ = ??
                action = ??
        action = action.cpu().numpy().squeeze(0) * self.act_limit
        return action

    def update(self, replay_buffer, batch_size):
        
        transitions = replay_buffer.sample(batch_size)
        s = torch.FloatTensor(np.array(transitions.s)).to(DEVICE)
        a = torch.FloatTensor(np.array(transitions.a)).to(DEVICE)
        r = torch.FloatTensor(np.array(transitions.r)).to(DEVICE).unsqueeze(-1)
        s2 = torch.FloatTensor(np.array(transitions.s2)).to(DEVICE)
        done = torch.FloatTensor(np.array(transitions.done)).to(DEVICE).unsqueeze(-1)

        
        a_scaled = a / self.act_limit

        # --- compute target Q value ---
        with torch.no_grad():
            a2, logp_a2, _ = ??
            a2_scaled = ??
            q1_t = ??
            q2_t = ??
            q_target_min = ??
            if AUTOMATIC_ENTROPY_TUNING:
                alpha = ??
            else:
                alpha = self.alpha
            # target y = r + gamma*(min_q - alpha * logp_a2)
            y = ??

        # --- Q losses ---
        q1_pred = ??
        q2_pred = ??
        q1_loss = ??
        q2_loss = ??

        self.q1_opt.zero_grad()
        ??.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        ??.backward()
        self.q2_opt.step()

        # --- Policy loss ---
        a_new, logp_new, _ = ??
        a_new_scaled = a_new * self.act_limit
        q1_new = ??
        q2_new = ??
        q_new_min =??

        if AUTOMATIC_ENTROPY_TUNING:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        policy_loss = ??

        self.policy_opt.zero_grad()
        ??.backward()
        self.policy_opt.step()

        # --- entropy (alpha) tuning ---
        if AUTOMATIC_ENTROPY_TUNING:
            alpha_loss = ??
            self.alpha_opt.zero_grad()
            ??.backward()
            self.alpha_opt.step()
            alpha = ??
        else:
            alpha = self.alpha

        # --- soft update targets ---
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": alpha if not isinstance(alpha, torch.Tensor) else alpha.item()
        }



env = gym.make(ENV_NAME)
env.reset(seed=SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

agent = SACAgent(env)
replay = ReplayBuffer(REPLAY_SIZE)

total_steps = 0
episode = 0
ep_return = 0
ep_len = 0

os.makedirs("sac_checkpoints", exist_ok=True)

state, _ = env.reset()
while total_steps < MAX_STEPS:
    if total_steps < START_STEPS:
        
        action = env.action_space.sample()
    else:
        action = agent.select_action(state, evaluate=False)

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    replay.push(state, action, reward, next_state, float(done))

    state = next_state
    ep_return += reward
    ep_len += 1
    total_steps += 1

    if done:
        episode += 1
        print(f"Episode {episode} | Steps {total_steps} | Return {ep_return:.2f} | Len {ep_len}")
        state, _ = env.reset()
        ep_return = 0
        ep_len = 0

    
    if total_steps >= UPDATE_AFTER and total_steps % UPDATE_EVERY == 0:
        for j in range(UPDATE_EVERY):
            if len(replay) < BATCH_SIZE:
                continue
            info = agent.update(replay, BATCH_SIZE)
            #print(info)


env.close()