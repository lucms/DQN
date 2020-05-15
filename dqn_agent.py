import random
from dqn import DQN
from replay_buffer import ReplayBuffer
import torch.optim
import torch


class DQNAgent:
    """
    Deep Q-Network Agent, containing a target DQN, a policy DQN and a Replay Buffer.
    Currently only supports Adam Optimizer.

    Parameters
    ----------
    state_dim : int
        Maximum number of transitions to store.
    action_dim : int
        Action space dimension. Currently only supports discrete actions.
    buffer_size : int, optional
        Maximum number of transitions to store in the agent's replay buffer. (Default is int(1e5))
    batch_size: int, optional
        Number of samples used to perform one iteration of optimization. (Default is 128)
    target_update_frequency : int, optional
        Frequency, in timesteps, of the target network update. (default is 1000)
    gamma: float, optional
        Q-value prediction parameter. (Default is 0.99)
    learning_rate: float, optional
        Learning rate for the agent's optimizer. (Default is 1-e3)
    weight_decay: float, optional
        Weight decay for the agent's optimizer. (Default is 1e-2)
    hidden_layers: tuple, optional
        Number of neurons of each hidden layer. Supports an arbitrary number of layers. (Default is (48,16))

    Attributes
    ----------
    state_dim : int
        Maximum number of transitions to store.
    action_dim : int
        Action space dimension. Currently only supports discrete actions.
    batch_size: int, optional
        Number of samples used to perform one iteration of optimization. (Default is 128)
    target_update_frequency : int, optional
        Frequency, in timesteps, of the target network update. (default is 1000)
    gamma: float, optional
        Q-value prediction parameter. (Default is 0.99)
    replay_buffer : ReplayBuffer
        Agent's replay buffer, used to store transitions.
    policy_dqn : DQN
        Agent's policy DQN.
    target_dqn : DQN
        Agent's target DQN, used for calculating expected Q-Values.
    optimizer : torch.optim.Adam
        Agent's optimization algorithm.
    """
    def __init__(self, state_dim, action_dim,
                 buffer_size=int(1e5),
                 batch_size=128,
                 target_update_frequency=1000,
                 gamma=0.99,
                 learning_rate=1e-3,
                 weight_decay=1e-2,
                 hidden_layers=(48, 16)):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.policy_dqn = DQN(state_dim, action_dim, hidden_layers=hidden_layers)
        self.target_dqn = DQN(state_dim, action_dim, hidden_layers=hidden_layers)
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

    def act(self, states, epsilon=0):
        """
        Get agent's action(s) for given state(s), which has epsilon probability of being random.

        Parameters
        ----------
        states : torch.Tensor
            State(s) to get the agent's action(s).
        epsilon : float, optional
            Probability of choosing a random action over the one with biggest predicted Q-Value.

        Returns
        ----------
        actions : torch.Tensor
            Actions chosen by the agent.
        """
        if random.random() > epsilon:
            states = torch.tensor(states, dtype=torch.float).unsqueeze(0)
            q_value = self.policy_dqn.forward(states)
            actions = q_value.max(1)[1].item()
        else:
            actions = random.randrange(self.action_dim)
        return actions

    def optimize(self, timestep):
        """
        Perform a single iteration of optimization on the agent's policy DQN, using batch_size samples and loss metric
        torch.nn.MSELoss.

        Parameters
        ----------
        timestep : int
            Timestep number, used for deciding of the target DQN should be updated.

        Returns
        ----------
        loss.item() : torch.Tensor
            Calculated loss for the actions in the batch.
        """
        states, actions, rewards, next_states, dones = zip(*self.replay_buffer.sample_transitions(self.batch_size))

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        q_values = self.policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss_function = torch.nn.MSELoss()
        loss = loss_function(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if timestep % self.target_update_frequency == 0:
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        return loss.item()


if __name__ == '__main__':
    # DQNAgent training example, on gym's CartPole-v0.
    import gym
    import numpy as np

    # Create the env, agent and get first state
    env = gym.make('CartPole-v0')
    agent = DQNAgent(action_dim=2, state_dim=4)
    state = env.reset()

    # Setting epsilon-greedy exploration parameters
    initial_eps = 1
    final_eps = 0.01
    eps_decay = (1 - 0.01) / 5000
    eps = initial_eps

    # Initializing report metrics
    num_episodes = 0
    returns = np.array([0])
    episode_reward = 0
    log_frequency = 1000

    # Training loop, for 100000 timesteps
    for ts in range(100000):
        # Get agent action and perform a step on the environment
        action = agent.act(state, epsilon=eps)
        env.render()
        next_state, reward, done, info = env.step(action)

        # Add S,A,R,S',d transition to the agent's buffer
        agent.replay_buffer.add_transition((state, action, reward, next_state, done))

        # Sum the timestep reward to the episode reward
        episode_reward += reward

        # Optimize the agent, if it has enough experiences stored
        if ts > agent.batch_size:
            agent.optimize(ts)

        # Update the epsilon value
        if ts < 5000:
            eps = eps - eps_decay

        # Log some training information
        if ts % log_frequency == 0:
            print("Last 100 episodes' average return:", returns[-100:].sum() / returns[-100:].shape[0])
            print('Num of episodes: ', num_episodes)

        # Reset the env and process log metrics if the episode is over
        if done:
            state = env.reset()
            num_episodes += 1
            returns = np.append(returns, episode_reward)
            episode_reward = 0

        else:
            state = next_state
    env.close()
