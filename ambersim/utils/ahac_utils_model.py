import torch
import torch.nn as nn
import numpy as np
from typing import Sequence


################################################## MODEL UTILS ##################################################

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_activation_func(activation_name):
    if activation_name.lower() == "tanh":
        return nn.Tanh()
    elif activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "elu":
        return nn.ELU()
    elif activation_name.lower() == "identity":
        return nn.Identity()
    else:
        raise NotImplementedError(
            "Activation func {} not defined".format(activation_name)
        )

    
################################################################################################################





#################################################### ACTOR UTILS ###############################################
    
class ActorDeterministicMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, units, activation: str, device="cuda:0"):
        super(ActorDeterministicMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + units + [action_dim]

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(get_activation_func(activation))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1])) #TODO : replace LayerNorm with jax/flax function

        self.actor = nn.Sequential(*modules).to(device)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.actor)

    def get_logstd(self):
        # return self.logstd
        return None

    def forward(self, observations, deterministic=False):
        return self.actor(observations)
    
################################################################################################################





################################################### CRITIC UTILS ###############################################


class DoubleCriticMLP(nn.Module):
    def __init__(self, obs_dim, units, activation: str, device="cuda:0"):
        super(DoubleCriticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + units + [1]

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(get_activation_func(activation))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic_1 = nn.Sequential(*modules).to(device)
        self.critic_2 = nn.Sequential(*modules).to(device)

        self.obs_dim = obs_dim

        print(self.critic_1)
        print(self.critic_2)

    def forward(self, observations):
        v1 = self.critic_1(observations)
        v2 = self.critic_2(observations)
        return torch.min(v1, v2)

    def predict(self, observations):
        """Different from forward as it returns both critic values estimates"""
        v1 = self.critic_1(observations)
        v2 = self.critic_2(observations)
        return torch.cat((v1, v2), dim=-1)

################################################################################################################
