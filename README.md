# Reinforcement Learning : Étude Comparative PPO vs SAC (Hopper-v5)

Ce projet, réalisé dans le cadre du module **IMDS5A** à **Polytech Clermont-Ferrand**, explore l’application de deux algorithmes majeurs d’apprentissage par renforcement profond — **Proximal Policy Optimization (PPO)** et **Soft Actor-Critic (SAC)** — pour le contrôle continu d’un robot monopode instable dans l’environnement **Hopper-v5 (MuJoCo)**.

---

## 🎯 Problématique

**Comment apprendre à un robot physiquement instable à sauter et courir de manière autonome sans tomber ?**

Ce projet compare :
- une approche **prudente et stable** (*PPO*),
- à une méthode **rapide et fortement exploratrice** (*SAC*),

afin d’identifier la stratégie la plus performante dans un environnement dynamique et chaotique.

---

## 🛠️ Algorithmes Implémentés

### 1. Proximal Policy Optimization (PPO)
- **Type** : On-policy  
- **Architecture** : Actor-Critic découplés (MLP 2×64 neurones)  
- **Mécanisme clé** : Clipping des mises à jour pour garantir une optimisation stable  


---

### 2. Soft Actor-Critic (SAC)
- **Type** : Off-policy  
- **Architecture** : Réseaux profonds (MLP 2×256 neurones) avec *Twin Critics*  
- **Mécanisme clé** : Maximisation de l’entropie pour encourager l’exploration  
- **Optimisation** : Réutilisation efficace des données via un *Replay Buffer*

---

## 💻 Installation et Utilisation

### Prérequis
- Python 3.8+
- PyTorch (support **CUDA recommandé**)
- Gymnasium `[mujoco]`
- MuJoCo

### Installation
```bash
pip install gymnasium[mujoco] torch numpy matplotlib
import gymnasium as gym

# Charger l'environnement
env = gym.make("Hopper-v5")

# Configurer les hyperparamètres
# Batch size : 64 (PPO), 256 (SAC)
# Lancer l'entraînement via le jupyter notebook 
```

## 👥 Auteurs

- **Ayman ZEJLI**
- **Loïc MAGNAN**

**Encadrant** : *Julien Hautot*  
**Institution** : Polytech Clermont-Ferrand — IMDS5A
