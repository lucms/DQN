# DQN
My implementation for the Deep Q-Network in [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) using PyTorch.

## How to Use
This project provides a Trainer for the implemented DQN. To use it, just create a [OpenAI Gym Env](https://gym.openai.com/envs/#classic_control) and pass it to the DQNTrainer, along with the model's desired hyperparamaters.

<script src="https://gist.github.com/lucms/a9eb531be904c23cd7a622f0cc2133c4#file-dqntrainer_example-py"></script>
To run a simple example, try

`python3 dqn_trainer.py`,

where a training session is performed on OpenAI Gym's CartPole-v0 environment.

