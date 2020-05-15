from dqn_agent import DQNAgent
import gym


class DQNTrainer:
    """
    Trainer for a Deep Q-Network Agent on giver environment. Currently only supports epsilon greedy exploration.

    Parameters
    ----------
    env : gym.Env
        Environment in which the training may occur.
    log_frequency : int, optional
        Frequency, in timesteps, to log training information. (Default is 1000)
    exploration : dict, optional
        Exploration algorithm to use in the training. Expects a dict in any of the following formats.
        1. For linear epsilon decay:
            {'algorithm': 'epsilon_greedy',
            'decay': 'linear',
            'initial_epsilon': 1.0,
            'final_epsilon': 0.01,
            'decay_timesteps': 1000}
        2. For exponential epsilon decay:
            {'algorithm': 'epsilon_greedy',
            'decay': 'exponential',
            'initial_epsilon': 1.0,
            'epsilon_decay': 0.995}
        (Default is as follows in 1.)
    **kwargs
        Optional keyword arguments for the DQNAgent's hyperparameters.

    Attributes
    ----------
    agent : DQNAgent
        Agent to be trained by the DQNTrainer on given environment.
    env : gym.Env
        Training environment for the agent.
    log_frequency : int
        Frequency, in timesteps, to log training information. (Default is 1000)
    exploration_config : dict
        Necessary parameters for setting the exploration algorithm, including which one to use.
    update_explo_param : function
        Updates the exploration parameter when called.


    """

    def __init__(self, env: gym.Env,
                 log_frequency=1000,
                 exploration=None,
                 **kwargs):

        self.log_frequency = log_frequency
        self.agent = DQNAgent(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0], **kwargs)
        self.env = env

        # Avoid mutable argument
        if exploration is None:
            exploration = {'algorithm': 'epsilon_greedy',
                           'decay': 'linear',
                           'initial_epsilon': 1.0,
                           'final_epsilon': 0.01,
                           'decay_timesteps': 1000}
        self.exploration_config = exploration

        # Parse the exploration dict to make the update_explo_param function
        if self.exploration_config['algorithm'] == 'epsilon_greedy':
            if self.exploration_config['decay'] == 'linear':
                update_term = (self.exploration_config['initial_epsilon'] - self.exploration_config[
                    'final_epsilon']) / self.exploration_config['decay_timesteps']
                self.update_explo_param = (lambda epsilon: epsilon - update_term if epsilon > self.exploration_config[
                    'final_epsilon'] else epsilon)

            elif self.exploration_config['decay'] == 'exponential':
                self.update_explo_param = (lambda epsilon: epsilon * self.exploration_config['epsilon_decay'])
        else:
            raise NotImplementedError

    def train(self, num_timesteps=100000, render=False):
        """
        Perform the training loop for num_timesteps duration.

        Parameters
        ----------
        num_timesteps : int, optional
            Number of timesteps to perform during the training session. (Default is 100000)
        render : bool, optional
            Whether to render the environment or not. (Default is False)
        """

        # Initialize log metrics
        episode_rewards = []
        episode_lengths = []
        episode_losses = []
        num_episodes = 0
        episode_reward = 0
        episode_length = 0
        episode_loss = 0

        # Get initial state and set initial epsilon
        state = self.env.reset()
        epsilon = self.exploration_config['initial_epsilon']

        # Perform the training loop num_timesteps times
        for timestep in range(num_timesteps):
            # Get agent action and perform a step on the environment
            action = self.agent.act(state, epsilon)
            if render:
                self.env.render()
            next_state, reward, done, info = self.env.step(action)

            # Add S,A,R,S',d transition to the agent's buffer
            self.agent.replay_buffer.add_transition((state, action, reward, next_state, done))

            # Optimize the agent, if it has enough experiences stored
            if timestep > self.agent.batch_size:
                loss = self.agent.optimize(timestep)
            else:
                loss = 0

            # Update the epsilon value and log metrics
            epsilon = self.update_explo_param(epsilon)
            episode_reward += reward
            episode_loss += loss
            episode_length += 1

            # Reset the environment if the episode has ended, logging more information
            if done:
                num_episodes += 1
                episode_rewards.append(episode_reward)
                episode_losses.append(episode_loss)
                episode_lengths.append(episode_length)

                episode_reward = 0
                episode_loss = 0
                episode_length = 0

                state = self.env.reset()

            else:
                state = next_state

            # Report the log metrics every log_frequency timesteps
            if timestep % self.log_frequency == 0:
                print('Num episodes: ', num_episodes)
                print('Num timesteps: ', timestep)
                print('Mean episode reward: ', sum(episode_rewards[-20:]) / min(20, max(1, len(episode_rewards))))
                print('Mean episode loss: ', sum(episode_losses[-20:]) / min(20, max(1, len(episode_losses))))
                print('Mean episode length: ', sum(episode_lengths[-20:]) / min(20, max(1, len(episode_lengths))))
                print('\n')


if __name__ == '__main__':
    # Simple DQNTrainer example
    trainer = DQNTrainer(gym.make('CartPole-v0'),
                         exploration={'algorithm': 'epsilon_greedy',
                                      'decay': 'linear',
                                      'initial_epsilon': 1.0,
                                      'final_epsilon': 0.01,
                                      'decay_timesteps': 10000},
                         learning_rate=1e-4,
                         gamma=0.99)

    trainer.train(render=True)
