from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional


class DQN(torch.nn.Module):
    """
    Deep Q-Network used to predict Q-Values. Currently only support ReLU activation function.

    Parameters
    ----------
    state_dim : int
        Maximum number of transitions to store.
    action_dim : int
        Action space dimension. Currently only supports discrete actions.
    hidden_layers: tuple, optional
        Number of neurons of each hidden layer. Supports an arbitrary number of layers. (The default is (128,64))

    Attributes
    ----------
    dqn : torch.nn.Sequential
        Deep Q-Network created from the parameters.
    """

    def __init__(self, state_dim, action_dim, hidden_layers=(128, 64)):
        super(DQN, self).__init__()

        # Create an ordered dict to store the layers
        nn_dict = OrderedDict()
        nn_dict['input_layer'] = nn.Linear(state_dim, hidden_layers[0])
        nn_dict['relu_1'] = nn.ReLU()

        for idx in range(len(hidden_layers)):
            next_idx = min(idx + 1, len(hidden_layers) - 1)
            nn_dict['hidden_layer_{}'.format(idx + 1)] = nn.Linear(hidden_layers[idx], hidden_layers[next_idx])
            nn_dict['relu_{}'.format(idx + 2)] = nn.ReLU()

        nn_dict['output_layer'] = nn.Linear(hidden_layers[-1], action_dim)

        # Compile the DQN based the nn_dict
        self.dqn = nn.Sequential(nn_dict)

    def forward(self, x):
        """
        Feedforward the given state(s) x in the DQN and obtain its predicted Q-Value(s).

        Parameters
        ----------
        x : torch.Tensor
            State(s) to propagate through the DQN.

        Returns
        -------
        dqn(x) : torch.Tensor
            Predicted Q-Value(s), produced by the DQN on x input.
        """
        return self.dqn(x)


if __name__ == '__main__':
    # Simple DQN test
    DQN = DQN(2, 2)
    q_val = DQN.forward(torch.randn(4, 2))
    print("Q-Value: \n", q_val)
