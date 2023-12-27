from torch import nn
from utils import make_layers
import torch
import logging
import einops

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, x, subnet, rnn):
        s, b, c, h, w = x.size()
        x = einops.rearrange(x, "s b c h w -> (s b) c h w")
        x = subnet(x)
        x = einops.rearrange(x, "(s b) c h w -> s b c h w", s=s, b=b)
        outs, ht = rnn(x, None, seq_len=s)
        return outs, ht

    def forward(self, x):
        x = x.transpose(0, 1)  # to b, s, c, h, w -> s, b, c, h, w
        hidden_states = []
        #logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            x, ht = self.forward_by_stage(
                x, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(ht)
        return tuple(hidden_states)

class Decoder(nn.Module):
    def __init__(self, subnets, rnns, forecast_steps):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.forecast_steps = forecast_steps
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, x, state, subnet, rnn):
        x, ht = rnn(x, state, seq_len=self.forecast_steps)
        s, b, c, h, w = x.size()
        x = einops.rearrange(x, "s b c h w -> (s b) c h w")
        x = subnet(x)
        x = einops.rearrange(x, "(s b) c h w -> s b c h w", s=s, b=b)
        return x

    def forward(self, hidden_states):
        x = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            x = self.forward_by_stage(x, hidden_states[i - 1],
                                        getattr(self, 'stage' + str(i)),
                                        getattr(self, 'rnn' + str(i)))
        x = x.transpose(0, 1)  # to b s 1 h w
        return x


class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class ED(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y
