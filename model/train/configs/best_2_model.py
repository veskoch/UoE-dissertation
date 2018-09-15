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
            num_layers=2,
            num_units=512,
            reduction_factor=3,
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=2,
            num_units=512,
            bridge=onmt.layers.CopyBridge(),
            attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3,
            residual_connections=False))

class NMTBig(onmt.models.SequenceToSequence):
    """Defines a bidirectional LSTM encoder-decoder model."""
    def __init__(self):
        super(NMTBig, self).__init__(
            source_inputter=onmt.inputters.WordEmbedder(
                vocabulary_file_key="source_words_vocabulary",
                embedding_size=512),
            target_inputter=onmt.inputters.WordEmbedder(
                vocabulary_file_key="target_words_vocabulary",
                embedding_size=512),
            encoder=onmt.encoders.BidirectionalRNNEncoder(
                num_layers=4,
                num_units=512,
                reducer=onmt.layers.ConcatReducer(),
                cell_class=tf.contrib.rnn.LSTMCell,
                dropout=0.3,
                residual_connections=False),
            decoder=onmt.decoders.AttentionalRNNDecoder(
                num_layers=4,
                num_units=512,
                bridge=onmt.layers.CopyBridge(),
                attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
                cell_class=tf.contrib.rnn.LSTMCell,
                dropout=0.3,
                residual_connections=False))

def model():
    return NMTBig()