import tensorflow as tf
import numpy as np

import magenta
from magenta.music import performance_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2
from magenta.pipelines.pipeline import _guarantee_dict
from magenta.music import sequences_lib

import os
import re

import copy

# Shortcut to chord symbol text annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
BEAT = music_pb2.NoteSequence.TextAnnotation.BEAT

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

class QuantizedSplitter(NoteSequencePipeline):
  """A Pipeline that splits quantized NoteSequences at regular intervals."""

  def __init__(self, hop_bars, metadata_df=None, name=None):
    """Creates a Splitter pipeline.

    Args:
      hop_bars: Hop size in bars that will be used to split a
          NoteSequence at regular intervals.
      name: Pipeline name.
    """
    super(QuantizedSplitter, self).__init__(name=name)
    self.hop_bars = hop_bars
    self.metadata_df = metadata_df

  def transform(self, note_sequence):
    seq_id = note_sequence.id
    val = self.metadata_df.loc[seq_id]['double_note_val']
    if val == 'Yes':
        multiplier=2
    else:
        multiplier=1
    return split_quantized_note_sequence(
        note_sequence, self.hop_bars, multiplier=multiplier)

def split_quantized_note_sequence(note_sequence, hop_bars, multiplier=1,
                        skip_splits_inside_notes=False):
  """Split one NoteSequence into many at specified intervals.

  If `hop_bars` is a scalar, this function splits a NoteSequence into
  multiple NoteSequences, all of fixed size. Each of the resulting NoteSequences is 
  shifted to start at time zero.

  If `hop_bars` is a list, the NoteSequence will be split at the specified bars.

  Args:
    note_sequence: The NoteSequence to split.
    hop_bars: The hop size, in bars, at which the NoteSequence will
        be split. Alternatively, this can be a Python list of bars at which 
        to split the NoteSequence.
    skip_splits_inside_notes: If False, the NoteSequence will be split at all
        hop positions, regardless of whether or not any notes are sustained
        across the potential split time, thus sustained notes will be truncated.
        If True, the NoteSequence will not be split at positions that occur
        within sustained notes.

  Returns:
    A Python list of NoteSequences.
  """

#   steps_per_quarter = note_sequence.quantization_info.steps_per_quarter
#   num = note_sequence.time_signature.numerator
#   denom = note_sequence.time_signature.denominator
#   bar_len = 4 * steps_per_quarter * num / denom

  steps_per_bar = int(sequences_lib.steps_per_bar_in_quantized_sequence(note_sequence))
  hop_size_quantized_steps = np.array(hop_bars) * steps_per_bar * multiplier
  hop_size_quantized_steps = hop_size_quantized_steps.tolist()

  notes_by_start_step = sorted(list(note_sequence.notes),
                               key=lambda note: note.quantized_start_step)
  note_idx = 0
  notes_crossing_split = []

  
  if isinstance(hop_size_quantized_steps, list):
    split_steps = sorted(hop_size_quantized_steps)
  else:
    split_steps = np.arange(
        hop_size_quantized_steps, note_sequence.total_quantized_steps, hop_size_quantized_steps)

  valid_split_steps = [0]

  for split_step in split_steps:
    # Update notes crossing potential split.
    while (note_idx < len(notes_by_start_step) and
           notes_by_start_step[note_idx].quantized_start_step < split_step):
      notes_crossing_split.append(notes_by_start_step[note_idx])
      note_idx += 1
    notes_crossing_split = [note for note in notes_crossing_split
                            if note.quantized_end_step > split_step]

    if not (skip_splits_inside_notes and notes_crossing_split):
      valid_split_steps.append(split_step)

  # Handle the final subsequence.
  if note_sequence.total_quantized_steps > valid_split_steps[-1]:
    valid_split_steps.append(note_sequence.total_quantized_steps)

  if len(valid_split_steps) > 1:
    return _extract_quantized_subsequences(note_sequence, valid_split_steps)
  else:
    return []

def _extract_quantized_subsequences(sequence, split_steps):
  """Extracts multiple subsequences from a quantized NoteSequence.

  Args:
    sequence: The quantized NoteSequence to extract subsequences from.
    split_times: A Python list of subsequence boundary steps. The first
        subsequence will start at `split_steps[0]` and end at `split_steps[1]`,
        the next subsequence will start at `split_steps[1]` and end at
        `split_steps[2]`, and so on with the last subsequence ending at
        `split_steps[-1]`.

  Returns:
    A Python list of new NoteSequence containing the subsequences of `sequence`.

  Raises:
    QuantizationStatusException: If the sequence has  NOT been quantized.
    ValueError: If there are fewer than 2 split steps, or the split steps are
        unsorted, or if any of the subsequences would start past the end of the
        sequence.
  """
  if not sequences_lib.is_quantized_sequence(sequence):
    raise sequences_lib.QuantizationStatusException(
        'Can only extract subsequences from quantized NoteSequence.')

  if len(split_steps) < 2:
    raise ValueError('Must provide at least a start and end step.')
  if any(t1 > t2 for t1, t2 in zip(split_steps[:-1], split_steps[1:])):
    raise ValueError('Split steps must be sorted.')

  subsequence = music_pb2.NoteSequence()
  subsequence.CopyFrom(sequence)

  subsequence.total_quantized_steps = 0

  del subsequence.notes[:]
  del subsequence.time_signatures[:]
  del subsequence.key_signatures[:]
  del subsequence.tempos[:]
  del subsequence.text_annotations[:]
  del subsequence.control_changes[:]
  del subsequence.pitch_bends[:]

  subsequences = [copy.deepcopy(subsequence)
                  for _ in range(len(split_steps) - 1)]

  # Extract notes into subsequences.
  subsequence_index = -1
  for note in sorted(sequence.notes, key=lambda note: note.quantized_start_step):
    if note.quantized_start_step < split_steps[0]:
      continue
    while (subsequence_index < len(split_steps) - 1 and
           note.quantized_start_step >= split_steps[subsequence_index + 1]):
      subsequence_index += 1
    if subsequence_index == len(split_steps) - 1:
      break
    subsequences[subsequence_index].notes.extend([note])
    subsequences[subsequence_index].notes[-1].quantized_start_step -= (
        split_steps[subsequence_index])
    subsequences[subsequence_index].notes[-1].quantized_end_step = min(
        note.quantized_end_step,
        split_steps[subsequence_index + 1]) - split_steps[subsequence_index]
    if (subsequences[subsequence_index].notes[-1].quantized_end_step >
        subsequences[subsequence_index].total_quantized_steps):
      subsequences[subsequence_index].total_quantized_steps = (
          subsequences[subsequence_index].notes[-1].quantized_end_step)

  # Extract time signatures, key signatures, tempos, and chord changes (beats
  # are handled below, other text annotations and pitch bends are deleted).
  # Additional state events will be added to the beginning of each subsequence.

  events_by_type = [
      sequence.time_signatures, sequence.key_signatures, sequence.tempos,
      [annotation for annotation in sequence.text_annotations
       if annotation.annotation_type == CHORD_SYMBOL]]
  new_event_containers = [
      [s.time_signatures for s in subsequences],
      [s.key_signatures for s in subsequences],
      [s.tempos for s in subsequences],
      [s.text_annotations for s in subsequences]
  ]

  for events, containers in zip(events_by_type, new_event_containers):
    previous_event = None
    subsequence_index = -1
    for event in sorted(events, key=lambda event: event.time):
      if event.time <= split_steps[0]:
        previous_event = event
        continue
      while (subsequence_index < len(split_steps) - 1 and
             event.time > split_steps[subsequence_index + 1]):
        subsequence_index += 1
        if subsequence_index == len(split_steps) - 1:
          break
        if previous_event is not None:
          # Add state event to the beginning of the subsequence.
          containers[subsequence_index].extend([previous_event])
          containers[subsequence_index][-1].time = 0
      if subsequence_index == len(split_steps) - 1:
        break
      # Only add the event if it's actually inside the subsequence (and not on
      # the boundary with the next one).
      if event.time < split_steps[subsequence_index + 1]:
        containers[subsequence_index].extend([event])
        containers[subsequence_index][-1].time -= split_steps[subsequence_index]
      previous_event = event
    # Add final state event to the beginning of all remaining subsequences.
    while subsequence_index < len(split_steps) - 2:
      subsequence_index += 1
      if previous_event is not None:
        containers[subsequence_index].extend([previous_event])
        containers[subsequence_index][-1].time = 0

  return subsequences



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