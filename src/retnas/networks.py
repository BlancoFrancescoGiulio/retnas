import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NetworkFactory:
    create_actor = staticmethod(lambda *args, **kwargs: SchedulerNetworkActorRNN(*args, **kwargs))
    create_critic = staticmethod(lambda *args, **kwargs: SchedulerNetworkCriticRNN(*args, **kwargs))

class SchedulerNetworkActorRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, use_packing=False, num_layers=1, type='gru'):
        super(SchedulerNetworkActorRNN, self).__init__()
        self.use_packing = use_packing
        self.type = type
        if type == 'gru':
            self.core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError('Unknown RNN type: {}'.format(type))
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.act = nn.Tanh()

    def forward(self, x, num_sjs):
        if self.use_packing:
            packed_x = pack_padded_sequence(x, num_sjs, batch_first=True, enforce_sorted=False)
            (packed_x, _) = self.core(packed_x)
            (x, _) = pad_packed_sequence(packed_x, batch_first=True)
        else:
            (x, _) = self.core(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.act(x)
        return x

class SchedulerNetworkCriticRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, use_packing=False, type='gru'):
        super(SchedulerNetworkCriticRNN, self).__init__()
        self.use_packing = use_packing
        self.type = type
        if type == 'gru':
            self.core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError('Unknown RNN type: {}'.format(type))
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, state, action, num_sjs):
        x = torch.cat((state, action), 2)
        if self.use_packing:
            packed_x = pack_padded_sequence(x, num_sjs, batch_first=True, enforce_sorted=False)
            (packed_x, _) = self.core(packed_x)
            (x, _) = pad_packed_sequence(packed_x, batch_first=True)
        else:
            (x, _) = self.core(x)
        x = x[torch.arange(x.shape[0]), num_sjs - 1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
