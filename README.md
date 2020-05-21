# DQN
My implementation for the Deep Q-Network in [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) using PyTorch.

## How to Use
This project provides a Trainer for the implemented DQN. To use it, just create a [OpenAI Gym Env](https://gym.openai.com/envs/#classic_control) and pass it to the DQNTrainer, along with the model's desired hyperparamaters. For a simple example:

```python
from dqn_trainer import DQNTrainer
import gym

trainer = DQNTrainer(gym.make('CartPole-v0'),
                     exploration={'algorithm': 'epsilon_greedy',
                                  'decay': 'linear',
                                  'initial_epsilon': 1.0,
                                  'final_epsilon': 0.01,
                                  'decay_timesteps': 10000},
                     learning_rate=1e-4,
                     gamma=0.99)

trainer.train(render=True)
 ```

## Next Features
The following features will be added to the project in the near future:
1. Saving and loading agent parameters;
2. Logging training data in csv;
3. Online plotting training data with Tensorboard.
