# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/encoders/rnn_encoder.py
"""Define RNN-based encoders."""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from c2nl.encoders.encoder import EncoderBase
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 bidirectional,
                 num_layers,
                 hidden_size,
                 dropout=0.0,
                 use_bridge=False,
                 use_last=True):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        # Saves preferences for layer
        self.nlayers = num_layers
        self.use_last = use_last

        self.rnns = nn.ModuleList()
        for i in range(self.nlayers):
            input_size = input_size if i == 0 else hidden_size * num_directions
            kwargs = {'input_size': input_size,
                      'hidden_size': hidden_size,
                      'num_layers': 1,
                      'bidirectional': bidirectional,
                      'batch_first': True}
            rnn = getattr(nn, rnn_type)(**kwargs)
            self.rnns.append(rnn)

        self.dropout = nn.Dropout(dropout)
        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            nl = 1 if self.use_last else num_layers
            self._initialize_bridge(rnn_type, hidden_size, nl)

    def count_parameters(self):
        params = list(self.rnns.parameters())
        if self.use_bridge:
            params = params + list(self.bridge.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, emb, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(emb, lengths)

        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Tensor.
            lengths, indices = torch.sort(lengths, 0, True)  # Sort by length (keep idx)
            packed_emb = pack(packed_emb[indices], lengths.tolist(), batch_first=True)
            _, _indices = torch.sort(indices, 0)  # Un-sort by length

        memory_bank, encoder_final = [], {'h_n': [], 'c_n': []}
        for i in range(self.nlayers):
            if i != 0:
                packed_emb = self.dropout(packed_emb)
                if lengths is not None:
                    packed_emb = pack(packed_emb, lengths.tolist(), batch_first=True)

            packed_emb, states = self.rnns[i](packed_emb)
            if isinstance(states, tuple):
                h_n, c_n = states
                encoder_final['c_n'].append(c_n)
            else:
                h_n = states
            encoder_final['h_n'].append(h_n)

            packed_emb = unpack(packed_emb, batch_first=True)[0] if lengths is not None else packed_emb
            if not self.use_last or i == self.nlayers - 1:
                memory_bank += [packed_emb[_indices]] if lengths is not None else [packed_emb]

        assert len(encoder_final['h_n']) != 0
        if self.use_last:
            memory_bank = memory_bank[-1]
            if len(encoder_final['c_n']) == 0:
                encoder_final = encoder_final['h_n'][-1]
            else:
                encoder_final = encoder_final['h_n'][-1], encoder_final['c_n'][-1]
        else:
            memory_bank = torch.cat(memory_bank, dim=2)
            if len(encoder_final['c_n']) == 0:
                encoder_final = torch.cat(encoder_final['h_n'], dim=0)
            else:
                encoder_final = torch.cat(encoder_final['h_n'], dim=0), \
                                torch.cat(encoder_final['c_n'], dim=0)

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        # TODO: Temporary hack is adopted to compatible with DataParallel
        # reference: https://github.com/pytorch/pytorch/issues/1591
        if memory_bank.size(1) < emb.size(1):
            dummy_tensor = torch.zeros(memory_bank.size(0),
                                       emb.size(1) - memory_bank.size(1),
                                       memory_bank.size(2)).type_as(memory_bank)
            memory_bank = torch.cat([memory_bank, dummy_tensor], 1)

        return encoder_final, memory_bank

    def _initialize_bridge(self,
                           rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
