import tensorflow as tf

import magenta
from magenta.music import performance_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2
from magenta.pipelines.pipeline import _guarantee_dict

import os
import re

import copy

# Shortcut to chord symbol text annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL

class NoteSequencePipeline(pipeline.Pipeline):
  """Superclass for pipelines that input and output NoteSequences."""

  def __init__(self, name=None):
    """Construct a NoteSequencePipeline. Should only be called by subclasses.

    Args:
      name: Pipeline name.
    """
    super(NoteSequencePipeline, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=music_pb2.NoteSequence,
        name=name)
    
class TransposerToC(NoteSequencePipeline):
    """Transposes all Note Sequences to C."""

    def __init__(self, name):
        """Creates a TranspositionToCPipeline.

        Args:
          name: Pipeline name.
        Returns:
            NoteSequence in C.
        """
        super(TransposerToC, self).__init__(name=name)
        print('INFO: Transposing all to C.')

    def transform(self, sequence):
        stats = dict([(state_name, statistics.Counter(state_name)) for state_name in
                      ['transpositions_generated']])

        key = sequence.key_signatures[0].key
        transposed = self._transpose(sequence, -key, stats)
        if transposed is not None:
            stats['transpositions_generated'].increment(1)
            self._set_stats(stats.values())
            return [transposed]
        
    def _transpose(self, ns, amount, stats):
        """Transposes a note sequence by the specified amount."""
        ts = copy.deepcopy(ns)
        for note in ts.notes:
            note.pitch += amount
        return ts

class TransposerToRange(NoteSequencePipeline):
  """Creates transposed versions of the input NoteSequence."""

  def __init__(self, 
                transposition_range, 
                min_pitch,
                max_pitch, 
                name):
    """Creates a TranspositionPipeline.

    Args:
      transposition_range: Collection of integer pitch steps to transpose.
      min_pitch: Integer pitch value below which notes will be considered
          invalid.
      max_pitch: Integer pitch value above which notes will be considered
          invalid.
      name: Pipeline name.
    """
    super(TransposerToRange, self).__init__(name=name)
    self._transposition_range = transposition_range
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch

    print('INFO: Transposition {}'.format(transposition_range))
    print('INFO: Transposition pipeline will ignore Key Signatures, Pitch Names and Chord Symbols.')

  def transform(self, sequence):
    stats = dict([(state_name, statistics.Counter(state_name)) for state_name in
                  ['skipped_due_to_range_exceeded',
                   'transpositions_generated']])

    transposed = []
    for amount in self._transposition_range:
      # Note that transpose is called even with a transpose amount of zero, to
      # ensure that out-of-range pitches are handled correctly.
      ts = self._transpose(sequence, amount, stats)
      if ts is not None:
        transposed.append(ts)

    stats['transpositions_generated'].increment(len(transposed))
    self._set_stats(stats.values())
    return transposed

  def _transpose(self, ns, amount, stats):
    """Transposes a note sequence by the specified amount."""
    ts = copy.deepcopy(ns)
    for note in ts.notes:
      if not note.is_drum:
        note.pitch += amount
        if note.pitch < self._min_pitch or note.pitch > self._max_pitch:
          stats['skipped_due_to_range_exceeded'].increment()
          return None
    return ts

class Reverser(NoteSequencePipeline):
    def __init__(self, active, name=None):
        """Creates a pipeline for reversing NoteSequences.

        Args:
        reverse: Reverse or nor. If False, returns the original. The use 
        of the flag is to prevent reversing of `train` and `test` datasets.
        """

        super(Reverser, self).__init__(name=name)
        self.active = active
        if active:
            print('INFO: Augmenting by reversing.')

    def transform(self, sequence):
        reversed_sequence = music_pb2.NoteSequence()
        reversed_sequence.CopyFrom(sequence)
        
        if not self.active:
            return [sequence]

        for note in reversed_sequence.notes:
            note.start_time = abs(note.start_time - sequence.total_time)
            note.end_time = abs(note.end_time - sequence.total_time)
        
        return [sequence, reversed_sequence]

class PerformanceExtractor(pipeline.Pipeline):
    """Extracts polyphonic tracks from a quantized NoteSequence."""

    def __init__(self, min_events, max_events, num_velocity_bins, name=None):
        super(PerformanceExtractor, self).__init__(
            input_type=music_pb2.NoteSequence,
            output_type=magenta.music.MetricPerformance,
            name=name)
        self._min_events = min_events
        self._max_events = max_events
        self._num_velocity_bins = num_velocity_bins

    def transform(self, quantized_sequence):
        performances, stats = magenta.music.extract_performances(
            quantized_sequence,
            num_velocity_bins=self._num_velocity_bins)
        self._set_stats(stats)

        return performances

class MetadataExtractor(pipeline.Pipeline):
    """Extracts polyphonic tracks from a quantized NoteSequence."""

    def __init__(self, metadata_df=None, attributes=None, name=None):

        self.metadata_df = metadata_df
        self.attributes = attributes

        print('INFO: Prepending metadata tokens for attributes: {}'.format(attributes))

        super(MetadataExtractor, self).__init__(
            input_type=music_pb2.NoteSequence,
            output_type=dict,
            name=name)

    def transform(self, sequence):
        meta = {}
        seq_id = sequence.id
        for attr in self.attributes:
            val = self.metadata_df.loc[seq_id][attr]
            if attr == 'segment': # strip digits
                val = re.sub("\d", "", val)
            if attr == 'key':
                val = re.sub("#", "sharp", val)
            if attr == 'time_sig':
                val = re.sub("/", "over", val)
            meta[attr] = val

        return [meta]

class ParserToText(pipeline.Pipeline):
    """Converts a Performance into a text sequence.
    
    Individual events become 'words' of A-Z 0-9 separated by space. 
    """
    
    def __init__(self, name=None):
        super(ParserToText, self).__init__(
            input_type={ 'MetricPerformance': magenta.music.MetricPerformance,
                         'metadata': dict },
            output_type=str,
            name=name)
        
    def transform(self, extracted):
        text_seq = []

        for key, val in extracted['metadata'].items():
            text_seq.append(val)

        for event in extracted['MetricPerformance']:
            if event.event_type == performance_lib.PerformanceEvent.NOTE_ON:
                text_seq.append('ON%s' % event.event_value)
            elif event.event_type == performance_lib.PerformanceEvent.NOTE_OFF:
                text_seq.append('OFF%s' % event.event_value)
            elif event.event_type == performance_lib.PerformanceEvent.TIME_SHIFT:
                text_seq.append('SHIFT%s' % event.event_value)
            else:
                raise ValueError('Unknown event type: %s' % event.event_type)
        
        return [' '.join(text_seq)]


def run_pipeline_text(pipeline,
                      input_iterator,
                      output_dir):
    """Runs a pipeline graph saving output to disk as text.
     
    Run the the pipeline on each input from the iterator one at a time.
    A file will be written to `output_dir` for each dataset name specified
    by the pipeline. pipeline.transform is called on each input and the
    results are aggregated into their correct datasets.

    The output type given by `pipeline.output_type` must be str.

    Args:
        pipeline: A Pipeline instance. `pipeline.output_type` must be a str.
        input_iterator: Iterates over the input data. Items returned by it are fed
            directly into the pipeline's `transform` method.
        output_dir: Path to directory where datasets will be written. Each dataset
            is a file whose name contains the pipeline's dataset name. If the
            directory does not exist, it will be created.
            
    Raises:
        ValueError: If any of `pipeline`'s output type is not str.
     
    """
    
    if isinstance(pipeline.output_type, dict):
        for name, type_ in pipeline.output_type.items():
            if type_ != str:
                raise ValueError(
                    'Pipeline "%s" must output %s type. '
                    'Output type was %s' % (name, str, type_))
    else:
         if type_ != str:
            raise ValueError(
                    'Pipeline "%s" must output %s type. '
                            'Output type was %s' % (name, str, pipeline.output_type))
    
    
    aggregated_outputs = dict(
        [(name, []) for name in pipeline.output_type_as_dict])
    total_inputs = 0
    total_outputs = 0
    stats = []
    
    output_names = pipeline.output_type_as_dict.keys()
    output_paths = [os.path.join(output_dir, name + '.txt')
                    for name in output_names]

    for path in output_paths:
        if os.path.exists(path):
            raise FileExistsError('File {} already exists. Please remove and try again.'
                                        .format(path))           

    writers = dict([(name, open(path, 'a'))
                  for name, path in zip(output_names, output_paths)])
    
    for input_object in input_iterator:
        total_inputs += 1

        for name, outputs in _guarantee_dict(
            pipeline.transform(input_object),
            list(output_names)[0]).items():            
            
            for output in outputs:
                writers[name].write(output + '\n')
                
            total_outputs += len(outputs)
        stats = statistics.merge_statistics(stats + pipeline.get_stats())
        if total_inputs % 5000 == 0:
            tf.logging.info('Processed %d inputs so far. Produced %d outputs.', 
                            total_inputs, total_outputs)
            statistics.log_statistics_list(stats, tf.logging.info)
    tf.logging.info('\n\nCompleted.\n')
    tf.logging.info('Processed %d inputs total. Produced %d outputs.',
                    total_inputs, total_outputs)
    statistics.log_statistics_list(stats, tf.logging.info)
    return aggregated_outputs

def build_dataset(pipeline_config, pipeline_graph_def):
    output_dir = pipeline_config['data_target_dir']
    
    print('INFO: Target {}.'.format(pipeline_config['data_target_dir']))
    print('INFO: Collated data sourced from {}.'.format(pipeline_config['data_source_dir']))

    for src_file in os.listdir(pipeline_config['data_source_dir']):
        if src_file.endswith('.tfrecord'):

            collection_name = os.path.splitext(src_file)[0]

            src_file_path = os.path.join(pipeline_config['data_source_dir'], src_file)

            print('\nINFO: Building {} dataset...'.format(collection_name))

            # Construct the pipeline graph
            pipeline_graph = pipeline_graph_def(
                collection_name = collection_name,
                config = pipeline_config
                )

            # Runs pipeline graph on a data source and writes output to dir
            run_pipeline_text(
                pipeline_graph,
                pipeline.tf_record_iterator(src_file_path,
                                            pipeline_graph.input_type),
                output_dir
                )

def build_vocab(pipeline_config, source_vocab_from=[]):
    """ This method stands on its own. Invoke after everything else.

    You need to change this method if you change the Encoding Format,
    or the Encoding Representation.
    """

    tokens = set()

    vocab = {
        'ON': (0, 127 + 1),
        'OFF': (0, 127 + 1),
        'SHIFT': (0, pipeline_config['steps_per_quarter'] * 4 + 1),
    }

    for action in vocab.keys():
        for val in range(vocab[action][0], vocab[action][1]):
            tokens.add(action + str(val))
    
    # Find tokens from external files,
    for source in source_vocab_from:
        path = os.path.join(pipeline_config['data_target_dir'], source)
        print("INFO: Collecting tokens from {}".format(path))
        with open(path, 'r') as f:
            for line in f.read().splitlines():
                for token in line.split():
                    tokens.add(token)
    
    vocab_path = pipeline_config['data_target_dir'] + 'vocab.txt'
    if os.path.exists(vocab_path):
        print("INFO: File {} exists. Removing. Rebuilding vocabulary.".format(vocab_path))
        os.remove(vocab_path)
    with open(vocab_path, 'a') as file:
        for token in tokens:
            file.write(token + '\n')
                
    print("INFO: Vocabulary built.")
    print('INFO: Tokens collected {}'.format(tokens))