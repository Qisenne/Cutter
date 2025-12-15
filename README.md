# Cutter
Learning to Compress Graphs via Dual Agents for Consistent Topological Robustness Evaluation

Our framework employs two deep reinforcement learning agents that collaboratively compress the original graph. The resulting compressed graph preserves the original network’s robustness profile, exhibiting similar degradation patterns when subjected to adversarial attacks.

---

Requirements The codebase is implemented in Python 3.12.7. package versions used for development are just below.  

pytorch 2.7.1  

tqdm 4.66.5  

networkx  3.4.2  

---

A simple way to improve training efficiency: by modifying the hyperparameter EPSILON_DECAY

