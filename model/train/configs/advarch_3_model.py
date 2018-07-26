"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt

class RNMTPlusSmall(onmt.models.SequenceToSequence):
  """Defines a model similar to the "Listen, Attend and Spell" model described
  in https://arxiv.org/abs/1508.01211.
  """
  def __init__(self):
    super(RNMTPlusSmall, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        encoder=onmt.encoders.RNMTPlusEncoder(
            num_layers=2,
            num_units=512,
            cell_class=tf.contrib.rnn.LayerNormBasicLSTMCell,
            dropout=0.3),
        decoder=onmt.decoders.MultiAttentionalRNNDecoder(
            num_layers=2,
            num_units=512,
            attention_layers=[0,1],
            attention_mechanism_class=tf.contrib.seq2seq.LuongMonotonicAttention,
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3,
            residual_connections=False))

def model():
    return RNMTPlusSmall()