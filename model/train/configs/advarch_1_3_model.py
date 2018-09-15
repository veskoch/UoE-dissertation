"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt

class ListenAttendSpell(onmt.models.SequenceToSequence):
  """Defines a model similar to the "Listen, Attend and Spell" model described
  in https://arxiv.org/abs/1508.01211.
  """
  def __init__(self):
    super(ListenAttendSpell, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        encoder=onmt.encoders.PyramidalRNNEncoder(
            num_layers=3,
            num_units=256,
            reduction_factor=1,
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3),
        decoder=onmt.decoders.MultiAttentionalRNNDecoder(
            num_layers=3,
            num_units=256,
            attention_layers=[0],
            attention_mechanism_class=tf.contrib.seq2seq.LuongMonotonicAttention,
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3,
            residual_connections=False))
            # Additionally, the cell state of this
            # decoder is not initialized from the encoder state (i.e. a
            # :class:`opennmt.layers.bridge.ZeroBridge` is imposed).

def model():
    return ListenAttendSpell()