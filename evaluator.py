"""Evaluation related classes and functions."""

from statistics import mean
import subprocess
import tempfile

import abc
import io
import re
import os
import six

import tensorflow as tf
import numpy as np

from opennmt.utils.misc import get_third_party_dir
from utils.text_seqs import TextSequence, TextSequenceCollection

from collections import Counter

import music21

@six.add_metaclass(abc.ABCMeta)
class ExternalEvaluator(object):
  """Base class for external evaluators."""

  def __init__(self, labels_file=None, output_dir=None):
    self._labels_file = labels_file
    self._summary_writer = None

    if output_dir is not None:
      self._summary_writer = tf.summary.FileWriterCache.get(output_dir)

  def __call__(self, step, predictions_path):
    """Scores the predictions and logs the result.

    Args:
      step: The step at which this evaluation occurs.
      predictions_path: The path to the saved predictions.
    """
    score = self.score(self._labels_file, predictions_path)
    if score is None:
      return
    if self._summary_writer is not None:
      self._summarize_score(step, score)
    self._log_score(score)

  def _summarize_value(self, step, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self._summary_writer.add_summary(summary, step)

  # Some evaluators may return several scores so let them the ability to
  # define how to log the score result.

  def _summarize_score(self, step, score):
    self._summarize_value(step, "score/{}".format(self.name()), score)

  def _log_score(self, score):
    tf.logging.info("%s evaluation score: %f", self.name(), score)

  @abc.abstractproperty
  def name(self):
    """Returns the name of this evaluator."""
    raise NotImplementedError()

  @abc.abstractmethod
  def score(self, labels_file, predictions_path):
    """Scores the predictions against the true output labels."""
    raise NotImplementedError()

class ROUGEEvaluator(ExternalEvaluator):
  """ROUGE evaluator based on https://github.com/pltrdy/rouge."""

  def __init__(self, labels_file=None, output_dir=None):
    try:
      import rouge  # pylint: disable=unused-variable
    except ImportError:
      raise ImportError("Please install the 'rouge' package, run 'pip install rouge==0.3.0'")
    super(ROUGEEvaluator, self).__init__(labels_file=labels_file, output_dir=output_dir)

  def name(self):
    return "ROUGE"

  def _summarize_score(self, step, score):
    self._summarize_value(step, "ROUGE-1/F1", score["rouge-1"]['f'])
    self._summarize_value(step, "ROUGE-1/Precision", score["rouge-1"]['p'])
    self._summarize_value(step, "ROUGE-1/Recall", score["rouge-1"]['r'])

    self._summarize_value(step, "ROUGE-2/F1", score["rouge-2"]['f'])
    self._summarize_value(step, "ROUGE-2/Precision", score["rouge-2"]['p'])
    self._summarize_value(step, "ROUGE-2/Recall", score["rouge-2"]['r'])

    self._summarize_value(step, "ROUGE-L/F1", score["rouge-l"]['f'])
    self._summarize_value(step, "ROUGE-L/Precision", score["rouge-l"]['p'])
    self._summarize_value(step, "ROUGE-L/Recall", score["rouge-l"]['r'])    


  def _log_score(self, score):
    tf.logging.info("Evaluation F1 Score: ROUGE-1 = %f; ROUGE-2 = %f; ROUGE-L = %s",
                    score["rouge-1"]['f'], score["rouge-2"]['f'], score["rouge-l"]['f'])
    tf.logging.info("Evaluation Precision Score: ROUGE-1 = %f; ROUGE-2 = %f; ROUGE-L = %s",
                    score["rouge-1"]['p'], score["rouge-2"]['p'], score["rouge-l"]['p'])
    tf.logging.info("Evaluation Recall Score: ROUGE-1 = %f; ROUGE-2 = %f; ROUGE-L = %s",
                    score["rouge-1"]['r'], score["rouge-2"]['r'], score["rouge-l"]['r'])

  def score(self, labels_file, predictions_path):
    from rouge import FilesRouge
    
    # Fix because otherwise I am reaching Pythons' recursion depth limit
    import sys
    sys.setrecursionlimit(10000)

    files_rouge = FilesRouge(predictions_path, labels_file)
    rouge_scores = files_rouge.get_scores(avg=True)
    print(rouge_scores)
    return rouge_scores


class BLEUEvaluator(ExternalEvaluator):
  """Evaluator calling multi-bleu.perl."""

  def _get_bleu_script(self):
    return "multi-bleu.perl"

  def name(self):
    return "BLEU"

  def score(self, labels_file, predictions_path):
    bleu_script = self._get_bleu_script()
    try:
      third_party_dir = get_third_party_dir()
    except RuntimeError as e:
      tf.logging.warning("%s", str(e))
      return None
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [os.path.join(third_party_dir, bleu_script), labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        return float(bleu_score)
    except subprocess.CalledProcessError as error:
      if error.output is not None:
        msg = error.output.strip()
        tf.logging.warning(
            "{} script returned non-zero exit code: {}".format(bleu_script, msg))
      return None


class BLEUDetokEvaluator(BLEUEvaluator):
  """Evaluator calling multi-bleu-detok.perl."""

  def _get_bleu_script(self):
    return "multi-bleu-detok.perl"

  def name(self):
    return "BLEU-detok"


class MusicScores(ExternalEvaluator):
    """Calculates the 1) Key Correlation Coefficient and 2) Key Tonal Certainty.
    Reports both 1) in absolute terms for predictions, and as relative between
    labels and predictions. 
    
    Key Correlation Coefficient: Shows how well this key fits the profile of a 
        piece in that key.
    Key Tonal Certainty: Provides a measure of tonal ambiguity for Key determined 
    with one of many methods.

    [This description is out of data.]
    
    """

    def name(self):
        return "MusicScores"

    def score(self, labels_file, predictions_path):
        """

        labels_file: Path to labels 
        predictions_path: Path to file with predictions.
        """

        labels = TextSequenceCollection(labels_file)
        predictions = TextSequenceCollection(predictions_path)

        datapoints = {
            'cc' : { 'labels' : [], 'predictions' : [] }, # Correlation Coefficient
            'tc' : { 'labels' : [], 'predictions' : [] }, # Tonal Certainty
            'key_name_matches' : 0.0,
            'key_mode_matches' : 0.0,
            'tempo' : { 'labels' : [], 'predictions' : [] }, # in bmp
            'duration' : { 'labels' : [], 'predictions' : [] } # in seconds
        }

        results = {
            'cc' : 0.0,
            'tc' : 0.0,
            'cc_dist' : 0.0,
            'tc_dist' : 0.0,
            'key_name_acc' : 0.0,
            'key_mode_acc' : 0.0,
            'tempo' : 0.0,
            'tempo_dist' : 0.0,
            'duration_dist' : 0.0
        }
        
        assert len(labels) == len(predictions)

        # Iterate in parallel over all (label, prediction) pairs
        for i in range(len(labels)):
            
            label_midi = labels.as_midi[i]
            prediction_midi = predictions.as_midi[i]
            
            with tempfile.NamedTemporaryFile(suffix='.midi') as temp_file:
                # call write on pretty_midi, and write to a temporary file
                label_midi.write(temp_file.name)
                mf = music21.midi.MidiFile()
                mf.open(temp_file.name)
                mf.read()
                mf.close()
                label_stream = music21.midi.translate.midiFileToStream(mf)

            with tempfile.NamedTemporaryFile(suffix='.midi') as temp_file:
                # call write on pretty_midi, and write to a temporary file
                prediction_midi.write(temp_file.name)
                mf = music21.midi.MidiFile()
                mf.open(temp_file.name)
                mf.read()
                mf.close()
                prediction_stream = music21.midi.translate.midiFileToStream(mf)           
            
            label_key = label_stream.analyze('key')
            prediction_key = prediction_stream.analyze('key')
            
            ###########################  
            ### COLLECT DATA-POINTS ###

            # Correlation Coefficient
            datapoints['cc']['labels'].append(label_key.correlationCoefficient)
            datapoints['cc']['predictions'].append(prediction_key.correlationCoefficient)

            # Tonal Certainty
            datapoints['tc']['labels'].append(label_key.tonalCertainty())
            datapoints['tc']['predictions'].append(prediction_key.tonalCertainty())

            # Is prediction in same key as labels?
            if label_key.tonic.name == prediction_key.tonic.name:
                datapoints['key_name_matches'] += 1

            # Is prediction same mode as labels?
            if label_key.mode == prediction_key.mode:
                datapoints['key_mode_matches'] += 1

            # Tempo (in bmp)
            try: # a few (not many) files fail, usually those with one note bar
                label_tempo = label_midi.estimate_tempo()
                prediction_tempo = prediction_midi.estimate_tempo()
                datapoints['tempo']['labels'].append(label_tempo)
                datapoints['tempo']['predictions'].append(prediction_tempo)
            except:
                pass

            # Duration
            datapoints['duration']['labels'].append(label_midi.get_end_time())
            datapoints['duration']['predictions'].append(prediction_midi.get_end_time())

        #############################  
        ### SUMMARIZE DATA-POINTS ###

        # Correlation Coefficient
        datapoints['cc']['labels'] = np.array(datapoints['cc']['labels'])
        datapoints['cc']['predictions'] = np.array(datapoints['cc']['predictions'])
        results['cc'] = np.mean(datapoints['cc']['predictions'])
        results['cc_dist'] = np.mean(np.abs(np.array(datapoints['cc']['labels']) - np.array(datapoints['cc']['predictions'])))

        # Tonal Certainty
        datapoints['tc']['labels'] = np.array(datapoints['tc']['labels'])
        datapoints['tc']['predictions'] = np.array(datapoints['tc']['predictions'])
        results['tc'] = np.mean(datapoints['tc']['predictions'])
        results['tc_dist'] = np.mean(np.abs(np.array(datapoints['tc']['labels']) - np.array(datapoints['tc']['predictions'])))

        # Key Accuracy
        results['key_name_acc'] = datapoints['key_name_matches'] / float(len(labels))
        results['key_mode_acc'] = datapoints['key_mode_matches'] / float(len(labels))

        # Tempo
        datapoints['tempo']['predictions'] = np.array(datapoints['tempo']['predictions'])
        datapoints['tempo']['labels'] = np.array(datapoints['tempo']['labels'])
        results['tempo'] = np.mean(datapoints['tempo']['predictions'])
        results['tempo_dist'] = np.mean(np.abs(datapoints['tempo']['labels'] - datapoints['tempo']['predictions']))

        # Duration
        datapoints['duration']['predictions'] = np.array(datapoints['duration']['predictions'])
        datapoints['duration']['labels'] = np.array(datapoints['duration']['labels'])
        results['duration_dist'] = np.mean(np.abs(datapoints['duration']['labels'] - datapoints['duration']['predictions']))

        return results

    def _summarize_score(self, step, score):
        self._summarize_value(step, "Correlation_Coefficient/Score", score['cc'])
        self._summarize_value(step, "Correlation_Coefficient/Distance", score['cc_dist'])

        self._summarize_value(step, "Tonal_Certainty/Score", score['tc'])
        self._summarize_value(step, "Tonal_Certainty/Distance", score['tc_dist'])

        self._summarize_value(step, "Key_Accuracy/Key_Name", score['key_name_acc'])
        self._summarize_value(step, "Key_Accuracy/Key_Mode", score['key_mode_acc'])

        self._summarize_value(step, "Tempo/Real_(bpm)", score['tempo'])
        self._summarize_value(step, "Tempo/Distance_(s)", score['tempo_dist'])

        self._summarize_value(step, "Duration/Distance_(s)", score['duration_dist'])

    def _log_score(self, score):
        tf.logging.info("Correlation_Coefficient/Score \t {}".format(score['cc']))
        tf.logging.info("Correlation_Coefficient/Distance \t {}".format(score['cc_dist']))

        tf.logging.info("Tonal_Certainty/Score \t {}".format(score['tc']))
        tf.logging.info("Tonal_Certainty/Distance \t {}".format(score['tc_dist']))

        tf.logging.info("Key_Accuracy/Key_Name \t {}".format(score['key_name_acc']))
        tf.logging.info("Key_Accuracy/Key_Mode \t {}".format(score['key_mode_acc']))

        tf.logging.info("Tempo/Real_(bpm) \t {}".format(score['tempo']))
        tf.logging.info("Tempo/Distance_(s) \t {}".format(score['tempo_dist']))

        tf.logging.info("Duration/Distance_(s) \t {}".format(score['duration_dist']))
        

def external_evaluation_fn(evaluators_name, labels_file, output_dir=None):
  """Returns a callable to be used in
  :class:`opennmt.utils.hooks.SaveEvaluationPredictionHook` that calls one or
  more external evaluators.

  Args:
    evaluators_name: An evaluator name or a list of evaluators name.
    labels_file: The true output labels.
    output_dir: The run directory.

  Returns:
    A callable or ``None`` if :obj:`evaluators_name` is ``None`` or empty.

  Raises:
    ValueError: if an evaluator name is invalid.
  """
  if evaluators_name is None:
    return None
  if not isinstance(evaluators_name, list):
    evaluators_name = [evaluators_name]
  if not evaluators_name:
    return None

  evaluators = []
  for name in evaluators_name:
    name = name.lower()
    if name == "bleu":
      evaluator = BLEUEvaluator(labels_file=labels_file, output_dir=output_dir)
    elif name == "bleu-detok":
      evaluator = BLEUDetokEvaluator(labels_file=labels_file, output_dir=output_dir)
    elif name == "rouge":
      evaluator = ROUGEEvaluator(labels_file=labels_file, output_dir=output_dir)
    else:
      raise ValueError("No evaluator associated with the name: {}".format(name))
    evaluators.append(evaluator)

  def _post_evaluation_fn(step, predictions_path):
    for evaluator in evaluators:
      evaluator(step, predictions_path)

  return _post_evaluation_fn
