# Rapport Technique : Étude Comparative des Algorithmes PPO et SAC pour le Contrôle Locomoteur

**Module :** IMDS5A - Ingénierie Mathématique et Data Science
**Établissement :** Polytech Clermont
**Auteurs :** Ayman ZEJLI & Loic MAGNAN
**Date :** 20 décembre 2025

---

## 1. Introduction et Problématique

L'Apprentissage par Renforcement (RL) s'est imposé comme une méthode incontournable pour la résolution de problèmes de contrôle continu complexes, tels que la robotique. Ce rapport se propose d'étudier et de comparer deux algorithmes majeurs de ce domaine, à savoir **Proximal Policy Optimization (PPO)** et **Soft Actor-Critic (SAC)**, appliqués à l'environnement de simulation **Hopper-v5**. Cet environnement, modélisant un robot monopode instable, constitue un banc d'essai rigoureux pour évaluer la capacité des algorithmes à acquérir une politique de contrôle robuste face à une dynamique chaotique.

**Problématique :** La question centrale qui guide cette étude est de déterminer quelle approche architecturale est la plus adaptée pour maîtriser un système dynamique instable comme le Hopper. Faut-il privilégier la **stabilité et la sécurité** de l'approche "On-Policy" (PPO), qui contraint les mises à jour de la politique, ou l'**efficacité et l'exploration** de l'approche "Off-Policy" (SAC), qui maximise l'entropie et réutilise les données passées ? Tout au long de ce développement, nous analyserons comment ces choix fondamentaux influencent directement la vitesse de convergence et la qualité de la politique finale.

## 2. Analyse Architecturale et Implémentation

L'analyse du code source développé pour cette étude met en lumière les divergences structurelles profondes entre les deux approches, chacune apportant une réponse technique différente à notre problématique.

### 2.1 Proximal Policy Optimization (PPO) : Architecture et Stabilité

L'implémentation de l'algorithme PPO repose sur une architecture découplée et synchrone, conçue pour éviter les mises à jour destructrices.

* **Classes `Policy` et `Value` :** Le code s'articule autour de deux réseaux de neurones distincts. La classe `Policy` modélise l'acteur et produit une distribution Gaussienne sur l'espace des actions. La classe `Value` estime la fonction de valeur $V(s)$.
* **Mécanisme de Clipping :** Le cœur de la stabilité de PPO réside dans sa fonction objectif "clippée". En limitant le ratio de probabilité entre la nouvelle et l'ancienne politique à l'intervalle $[1 - \epsilon, 1 + \epsilon]$ (avec $\epsilon = 0.2$), l'algorithme s'interdit les changements brusques de comportement. L'objectif maximisé est :

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t) ]
$$

Ce mécanisme garantit que la mise à jour de la politique reste conservatrice.

* **Réponse à la problématique :** Ce choix architectural favorise la sécurité. Sur un robot instable, cela signifie que l'agent ne "désapprendra" pas brutalement une marche acquise, mais progressera lentement.

### 2.2 Soft Actor-Critic (SAC) : Maximisation de l'Entropie

L'algorithme SAC propose une architecture plus complexe, orchestrée par la classe `SACAgent`, visant à extraire le maximum d'information de chaque interaction.

* **`ReplayBuffer` et Off-Policy :** Contrairement à PPO, SAC stocke les transitions passées dans un tampon mémoire pour les réutiliser. Cela permet une efficacité d'échantillonnage bien supérieure.
* **`GaussianPolicy` et Entropie :** L'agent utilise une politique stochastique qui maximise non seulement la récompense, mais aussi l'entropie. L'objectif global est de maximiser :

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_{\pi}} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]
$$

Le terme $\alpha \mathcal{H}(\pi(\cdot|s_t))$ force l'agent à explorer des stratégies variées et empêche la convergence prématurée.

* **`QNetwork` (Twin Critics) :** L'utilisation de deux critiques ($Q_1, Q_2$) et la prise de leur minimum pour la cible d'apprentissage corrigent les biais d'optimisme.
* **Réponse à la problématique :** Cette architecture favorise la découverte rapide de solutions. L'entropie empêche l'agent de rester figé dans une posture statique (optimum local fréquent sur Hopper) et le pousse à trouver une marche dynamique.

## 3. Protocole Expérimental et Hyperparamètres

Les expériences ont été menées sur l'environnement Hopper-v5 avec des configurations d'hyperparamètres spécifiques à chaque algorithme pour assurer une comparaison équitable de leurs capacités optimales.

* **PPO :** Entraîné sur **1000 époques** (2048 pas/époque). Taux d'apprentissage de $3 \times 10^{-4}$, clipping $\epsilon=0.2$. Ce grand nombre d'époques est nécessaire pour compenser la prudence de l'algorithme.
* **SAC :** Entraîné sur **300 000 steps** au total. Batch size de 256, capacité du buffer de $10^6$. L'ajustement automatique de l'entropie est activé pour réguler dynamiquement l'exploration.

## 4. Analyse des Performances et Résultats

L'analyse des courbes de récompense permet de trancher sur l'efficacité relative des deux méthodes face à notre problématique.

### 4.1 Analyse de PPO : Progression et Stabilité

![Performance PPO](ppo_800_epochs.png)
*Figure 1 : Évolution de la récompense moyenne sur 800 époques pour l'algorithme PPO.*

La courbe d'apprentissage de PPO (Figure 1) illustre une progression régulière et quasi-monotone. Partant d'un comportement aléatoire, l'agent améliore progressivement sa politique pour atteindre une marche stable (reward ~3500). Cette progression linéaire valide l'hypothèse de stabilité : l'approche "Trust Region" empêche les chutes de performance, mais impose un temps d'apprentissage long.

### 4.2 Analyse de SAC : Efficacité d'Échantillonnage

![Performance SAC](sac_plot.png)
*Figure 2 : Évolution de la récompense cumulée sur 300 000 steps pour l'algorithme SAC.*

L'algorithme SAC (Figure 2) démontre une dynamique radicalement différente. Après une brève phase initiale, les performances explosent. Dès 50 000 steps, l'agent atteint des niveaux de récompense comparables à ceux obtenus par PPO en fin d'entraînement. Cette efficacité valide l'hypothèse de l'exploration par l'entropie : le `ReplayBuffer` et la maximisation de l'entropie permettent de trouver la solution optimale avec un ordre de grandeur de moins d'interactions.

### 4.3 Défis Computationnels et Instabilité

Au cours de nos expérimentations, nous avons été confrontés à des défis significatifs liés au réglage du nombre d'époques, illustrant la difficulté de maintenir un équilibre entre exploration et exploitation.

* **Phénomène d'Effondrement (Catastrophic Forgetting) :** Lors de tests prolongés (ex: 5000 époques), nous avons observé que la récompense, après avoir atteint un pic optimal, avait tendance à s'effondrer brutalement. L'agent, en continuant d'explorer ou à cause de mises à jour trop agressives sur des données bruitées, "oubliait" sa politique de marche stable pour adopter des comportements erratiques.
* **Réglage de la Durée d'Entraînement :** Des essais trop courts (300 époques) ne permettaient pas à PPO de converger vers une marche fluide. À l'inverse, un entraînement trop long augmentait le risque de divergence.
* **Compromis :** Nous avons finalement retenu **1000 époques** comme le point d'équilibre optimal, permettant d'atteindre le plateau de performance (~3500 reward) avant que l'instabilité ne dégrade la politique. Cela souligne que "plus d'entraînement" n'est pas toujours synonyme de "meilleure performance" en RL, surtout sur des dynamiques instables comme le Hopper.

## 5. Conclusion

Pour conclure sur notre problématique initiale concernant le choix entre stabilité (PPO) et efficacité (SAC) pour le contrôle du Hopper-v5 :

**L'algorithme Soft Actor-Critic (SAC) se révèle supérieur pour cette tâche.**

Bien que PPO offre une garantie de stabilité appréciable et une simplicité d'implémentation (classes découplées), il souffre d'une inefficacité d'échantillonnage qui le rend lent. SAC, grâce à son architecture Off-Policy et à la régularisation par l'entropie, parvient à maîtriser la dynamique instable du Hopper beaucoup plus rapidement. Il réussit à transformer l'instabilité du système en opportunité d'exploration, là où PPO tente de la contraindre.

Ainsi, pour une application réelle où le temps de simulation ou d'interaction est coûteux, **SAC est le choix préconisé**. PPO reste une alternative pertinente uniquement si la simplicité de mise en œuvre prime sur la performance brute ou si la stabilité monotone est une contrainte absolue de sécurité.

---

### Bibliographie

1. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv.
2. Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL*. arXiv.
3. Documentation Gymnasium & MuJoCo.
