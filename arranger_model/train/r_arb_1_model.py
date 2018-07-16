"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt

class NMTSmall_plain(onmt.models.SequenceToSequence):
    """Defines a small unidirectional LSTM encoder-decoder model."""

    def __init__(self):
        super(NMTSmall_plain, self).__init__(
            source_inputter=onmt.inputters.WordEmbedder(
                vocabulary_file_key="source_words_vocabulary",
                embedding_size=64),
            target_inputter=onmt.inputters.WordEmbedder(
                vocabulary_file_key="target_words_vocabulary",
                embedding_size=64),
            encoder=onmt.encoders.UnidirectionalRNNEncoder(
                num_layers=2,
                num_units=512,
                cell_class=tf.contrib.rnn.LSTMCell,
                dropout=0.3,
                residual_connections=False),
            decoder=onmt.decoders.RNNDecoder(
                num_layers=2,
                num_units=512,
                bridge=onmt.layers.CopyBridge(),
                cell_class=tf.contrib.rnn.LSTMCell,
                dropout=0.3,
                residual_connections=False))

def model():
    return NMTSmall_plain()