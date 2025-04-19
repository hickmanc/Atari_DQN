Atari_DQN

A minimal implementation of Deep Q-Networks (DQN) for playing Atari games using PyTorch. The agent learns to play directly from raw pixel inputs (like a human would, just looking at the screen) using key techniques such as experience replay and target networks to stabilize training.

This project is heavily influenced by the codebase from Deep Reinforcement Learning Hands-On (Third Edition). However, this implementation makes a key improvement: it includes a GPU-accelerated experience replay buffer, which avoids the overhead of repeatedly copying data between CPU and GPU during training.

Currently, the implementation has only been verified to solve the Pong environment. It is designed for clarity and conceptual understanding, and purposefully omits advanced DQN extensions (e.g., Double DQN, Dueling Networks, Prioritized Replay) to keep the core algorithm simple and readable.