# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/decoder.py
import torch
import torch.nn as nn

from c2nl.decoders.decoder import RNNDecoderBase
from c2nl.utils.misc import aeq


class RNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.
    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [batch x len x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (batch x src_len x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Tensor): final hidden state from the decoder.
            decoder_outputs (Tensor): output from the decoder (after attn)
                         `[batch x tgt_len x hidden]`.
            attns (Tensor): distribution over src at each tgt
                        `[batch x tgt_len x src_len]`.
        """
        # Initialize local and return variables.
        attns = {}

        emb = tgt
        assert emb.dim() == 3

        coverage = state.coverage

        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_batch, tgt_len, _ = tgt.size()
        output_batch, output_len, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        if self.attn is not None:
            decoder_outputs, p_attn, coverage_v = self.attn(
                rnn_output.contiguous(),
                memory_bank,
                memory_lengths=memory_lengths,
                coverage=coverage,
                softmax_weights=False
            )
            attns["std"] = p_attn
        else:
            decoder_outputs = rnn_output.contiguous()

        # Update the coverage attention.
        if self._coverage:
            if coverage_v is None:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
            else:
                coverage = coverage + coverage_v \
                    if coverage is not None else coverage_v
            attns["coverage"] = coverage

        decoder_outputs = self.dropout(decoder_outputs)
        # Run the forward pass of the copy attention layer.
        if self._copy and not self._reuse_copy_attn:
            _, copy_attn, _ = self.copy_attn(decoder_outputs,
                                             memory_bank,
                                             memory_lengths=memory_lengths,
                                             softmax_weights=False)
            attns["copy"] = copy_attn
        elif self._copy:
            attns["copy"] = attns["std"]

        return decoder_final, decoder_outputs, attns
