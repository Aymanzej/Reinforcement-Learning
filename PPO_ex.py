# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:36:26 2025

@author: julien.hautot
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# --- Hyperparamètres ---
env_name = "Hopper-v5"
gamma = 0.99                   
lr = 1e-3                      
clip_eps = 0.2 # Coefficient de clip PPO 
epochs = 30                  
steps_per_epoch = 1000         
batch_size = 64                
entropy_coef = 0.01 # empeche le collapsing de la politique            

device = "cuda" if torch.cuda.is_available() else "cpu"


# --- Environnement ---
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# --- Réseau Policy (Gaussian) ---
# Actor
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))  

    # Création de la distribution gaussienne 
    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std)
        return mean, std


# --- Réseau Value ---
# Critic
class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # retourne Vϕ(s) ie estime la valeur de l'et
        )

    def forward(self, x):
        return self.net(x)


# --- Initialisation ---
policy = Policy().to(device)
value = Value().to(device)
optimizer_policy = optim.Adam(policy.parameters(), lr=lr)
optimizer_value = optim.Adam(value.parameters(), lr=lr)

# --- Fonction pour générer trajectoires ---
def collect_trajectories():
    obs = env.reset()[0]
    obs_list, act_list, rew_list, logp_list, val_list, done_list = [], [], [], [], [], []
    for _ in range(steps_per_epoch):
        obs_tensor = torch.FloatTensor(obs).to(device)
        mu, std = policy.forward(obs_tensor)
        dist = torch.distributions.Normal(mu, std)
        act = dist.sample().item()
        logp = dist.log_prob(act)
        val = value.forward(obs_tensor)

        # Scale action to environment
        act_clamped = torch.tanh(act) # Environment est déjà entre -1,1 et tanh envoi les actions aussi sur -1,1
        next_obs, rew, terminated, truncated, _ = env.step(act_clamped.cpu().detach().numpy())
        done = terminated or truncated

        # Stockage
        obs_list.append(obs)
        act_list.append(act.detach())
        rew_list.append(rew)
        logp_list.append(logp.detach())
        val_list.append(val.detach())
        done_list.append(done)

        obs = next_obs
        if done:
            obs = env.reset()[0]

    return obs_list, act_list, rew_list, logp_list, val_list, done_list

# --- Fonction pour calculer les avantages (GAE) et les retours ---
def compute_advantages(rews, vals, dones):
    advs, gae = [], 0
    vals = vals + [0]  # V(s_T) = 0 
    # Parcours des reawrds de l'épisode à l'envers
    for t in reversed(range(len(rews))):
        # Calcul du TD error : delta = r + gamma * V(s_{t+1}) - V(s_t)
        delta = rews[t] + gamma * vals[t + 1] * (1 - dones[t]) - vals[t]
        gae = delta + gamma * 0.95 * (1 - dones[t]) * gae # Avantage qui est généralisé
        advs.insert(0, gae)  
    returns = [advs[i] + vals[i] for i in range(len(advs))]

    return torch.FloatTensor(advs).to(device), torch.FloatTensor(returns).to(device)

# --- Entraînement ---
for epoch in range(epochs):
    obs_list, act_list, rew_list, logp_list, val_list, done_list = collect_trajectories()
    advs, returns = compute_advantages(rew_list, val_list, done_list)
    
    obs_tensor = torch.FloatTensor(obs_list).to(device)
    act_tensor = torch.stack(act_list).to(device)
    old_logp_tensor = torch.stack(logp_list).to(device)
    
    # --- Mise à jour PPO ---
    for _ in range(10):  # 10 mini-epochs
        idx = np.random.permutation(len(obs_list))
        for start in range(0, len(obs_list), batch_size):
            end = start + batch_size
            batch_idx = idx[start:end]

            obs_b = obs_tensor[batch_idx]
            act_b = act_tensor[batch_idx]
            adv_b = advs[batch_idx]
            ret_b = returns[batch_idx]
            old_logp_b = old_logp_tensor[batch_idx]

            # Policy
            mu, std = policy(obs_b)
            dist = torch.distributions.Normal(mu, std)
            logp = dist.log_prob(act_b).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()
            ratio = (logp - old_logp_b).exp()
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            loss_policy = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            # Value
            val_pred = value(obs_b)
            loss_value = torch.nn.functional.mse_loss(val_pred, ret_b)

            # Backprop
            optimizer_policy.zero_grad()
            loss_policy.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            loss_value.backward()
            optimizer_value.step()


    # --- Stats ---
    print(f"Epoch {epoch+1} | Avg Reward = {np.mean(rew_list):.2f}")