"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt

class NMTMedium(onmt.models.SequenceToSequence):
    """Defines a medium-sized bidirectional LSTM encoder-decoder model."""
    def __init__(self):
        super(NMTMedium, self).__init__(
            source_inputter=onmt.inputters.WordEmbedder(
                vocabulary_file_key="source_words_vocabulary",
                embedding_size=64),
            target_inputter=onmt.inputters.WordEmbedder(
                vocabulary_file_key="target_words_vocabulary",
                embedding_size=64),
            encoder=onmt.encoders.BidirectionalRNNEncoder(
                num_layers=2,
                num_units=1024,
                reducer=onmt.layers.ConcatReducer(),
                cell_class=tf.contrib.rnn.LSTMCell,
                dropout=0.3,
                residual_connections=False),
            decoder=onmt.decoders.AttentionalRNNDecoder(
                num_layers=2,
                num_units=1024,
                bridge=onmt.layers.CopyBridge(),
                attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
                cell_class=tf.contrib.rnn.LSTMCell,
                dropout=0.3,
                residual_connections=False))

def model():
    return NMTMedium()