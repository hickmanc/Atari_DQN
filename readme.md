Atari_DQN

An implementation of Deep Q-Networks (DQN) for playing Atari 2600 games using PyTorch. This project is inspired by the seminal work of Mnih et al. in “Playing Atari with Deep Reinforcement Learning”, and heavily influenced by the codebase from Deep Reinforcement Learning Hands-On (Third Edition).

Overview

This repository contains a minimal, educational implementation of the DQN algorithm applied to Atari games. The agent learns to play directly from raw pixel inputs using key techniques such as experience replay and target networks to stabilize training.

While the code is largely inspired by the Packt book, this implementation makes a key improvement: it includes a GPU-accelerated experience replay buffer, which avoids the overhead of repeatedly copying data between CPU and GPU during training. This can lead to a meaningful speedup, particularly when training on high-end GPUs.

Currently, the implementation has only been verified to solve the Pong environment. It is designed for clarity and conceptual understanding, and purposefully omits advanced DQN extensions (e.g., Double DQN, Dueling Networks, Prioritized Replay) to keep the core algorithm simple and readable.