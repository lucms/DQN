import random


class ReplayBuffer:
    """
    Data structure used to store transition tuples.

    Parameters
    ----------
    buffer_size : int
        Maximum number of transitions to store.

    Attributes
    ----------
    replay_buffer : list
        List in which the transition tuples are stored.
    buffer_size : int
        Maximum number of transitions to store.
    """

    def __init__(self, buffer_size):
        self.replay_buffer = []
        self.memory_size = buffer_size

    def add_transition(self, transition):
        """
        Appends given transition tuple to the replay_buffer and discards the first transition if it is full.

        Parameters
        ----------
        transition : tuple
            The transition tuple to store.
        """
        if len(self.replay_buffer) < self.memory_size:
            self.replay_buffer.append(transition)
        else:
            self.replay_buffer[:-1] = self.replay_buffer[1:]
            self.replay_buffer[-1] = transition

    def sample_transitions(self, num_samples):
        """
        Randomly sample batch_size transitions from the replay_buffer.

        Parameters
        ----------
        num_samples : int
            The number of transitions to sample.
        """
        return random.sample(self.replay_buffer, num_samples)

    def get_replay_buffer_size(self):
        """
        Get the replay buffer's size.

        Returns
        -------
        memory size : int
            Maximum number of transitions to store.
        """
        return self.memory_size


if __name__ == '__main__':
    # Simple replay buffer testing
    replay_buffer = ReplayBuffer(buffer_size=4)
    for _ in range(1000):
        replay_buffer.add_transition((0, 10, 100, 1000))
        replay_buffer.add_transition((1, 11, 101, 1001))
        replay_buffer.add_transition((2, 12, 102, 1002))
        sampled_transitions = replay_buffer.sample_transitions(2)
