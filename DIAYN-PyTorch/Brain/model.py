from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits

class StateMI(nn.Module):
    def __init__(self, n_object_states, n_robot_states, hidden):
        """
        Mutual Information Estimation Network in PyTorch
        """
        super(StateMI, self).__init__()

        # Define the hidden layer dimensions
        self.hidden = hidden
        # Define the network layers
        self.fc1 = nn.Linear(n_object_states, int(self.hidden/2))  # First hidden layer for x
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(n_robot_states, int(self.hidden/2))  # First hidden layer for y
        init_weight(self.fc2)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(int(self.hidden/2), 1)  # Output layer
        init_weight(self.fc3, initializer="xavier uniform")
        self.fc3.bias.data.zero_()

    def forward(self, obj_state, robot_state):
        """
        Forward pass through the network.
        
        """

        # Shuffle and concatenate
        x_in = robot_state
        y_in = obj_state
        # print("x_in", x_in.shape)
        # print("y_in", y_in.shape)

        y_shuffle = y_in[torch.randperm(y_in.size(0))]  # Shuffle y

        # Concatenate the observations
        x_conc = torch.cat([x_in, x_in], dim=-2)  # Concatenate along the second last dimension
        y_conc = torch.cat([y_in, y_shuffle], dim=-2)
        # print("x_conc", x_conc.shape)
        # print("y_conc", y_conc.shape)
        # Forward pass through the first layer
        layerx = F.relu(self.fc1(x_conc))  # First layer for x
        # print("layerx", layerx.shape)
        layery = F.relu(self.fc2(y_conc))  # First layer for y
        # print("layerx", layerx.shape)
        layer2 = F.relu(layerx + layery)  # Combine the outputs
        # print("layer2", layer2.shape)
        # Output layer
        output = self.fc3(layer2)
        # print("output", output.shape)
        output = torch.tanh(output)  # Apply tanh activation to the output
        # Split output into T_xy and T_x_y predictions
        # print("output", output.shape)
        N_samples = x_in.size(1)
        # print("N_samples", N_samples)
        T_xy = output[:, : N_samples, :]
        T_x_y = output[:, N_samples:, :]
        # print("T_xy", T_xy.shape)
        # print("T_x_y", T_x_y.shape)
        # Compute the negative loss (maximize loss == minimize -loss)
        mean_exp_T_x_y = torch.mean(torch.exp(T_x_y), dim=-2)
        neg_loss = -(torch.mean(T_xy, dim=-2) - torch.log(mean_exp_T_x_y))
        # Return the final MI loss
        return neg_loss

class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)


class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        if torch.isnan(mu).any():

            print("NaN in mu:", mu)
            print("states", states)
            raise ValueError("mu has NaNs")

        log_std = self.log_std(x)

        if torch.isnan(log_std).any():
            print("NaN in log_std before clamp:", log_std)
            print("states", states)
            raise ValueError("log_std has NaNs")
        
        std = torch.exp(log_std.clamp(min=-20, max=2))

        if torch.isnan(std).any():
            print("NaN in std after exp:", std)
            print("states", states)
            raise ValueError("std has NaNs")
        
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states):
        dist = self(states)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob
