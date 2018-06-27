import magenta
from magenta.models.performance_rnn import performance_model
import tensorflow as tf

"""Constants for music processing."""

SOURCE_XML_DIR = "/Users/vesko/Google Drive/Docs/Education/Edinburgh/Classes/DISS/Data/MSc 2018 Research/Preprocessed Dataset"

COLLATED_NOTE_SEQ_DIR = "./data/note_seqs/"
ENCODED_SEQ_EXMPL_DIR = "./data/seq_examples/"

# INPUT_DIR = "./tmp/raw_xml/"
# TFRECORD_FILE = "./tmp/test.tfrecord"
# OUTPUT_DIR = "./tmp/sequence_examples/"
EVAL_RATIO = 0.1
TEST_RATIO = 0.1

CONFIG = 'performance'

default_configs = {
    'performance': performance_model.PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='performance',
            description='Performance RNN'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.PerformanceOneHotEncoding()),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001))
}