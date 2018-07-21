"""Example of a translation client."""

from __future__ import print_function

import argparse
import tempfile

# Model Serving
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations

# Processing Pipelines
import magenta
from magenta.protobuf import music_pb2
from magenta.pipelines import dag_pipeline
from magenta.pipelines.pipeline import _guarantee_dict
from magenta.pipelines.note_sequence_pipelines import Quantizer

from utils.text_seqs import TextSequence
from utils.arranger_pipelines import PerformanceExtractor, MetadataExtractor, ParserToText


# CONSTANTS
STEPS_PER_QUARTER = 4

MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127

MIN_EVENTS = 1
MAX_EVENTS = 9999999


class Enhancer():
  """Use to run inference on a trained model.

  On initialization, the object starts a server which can be queried
  with the predict() method. The method takes in the user input
  as a `NoteSequence` proto and returns a MIDI produced by the model.
  """

  def __init__(self, model_name, timeout=10.0, host="localhost", port=9000):
    """[summary]
    
    Keyword Arguments:
      model_name {str} -- [Model name. Must match name given to Tensorflow Serving.]
      timeout {float} -- [Requested timeout.] (default: {10.0})
      host {str} -- [Server host for the model.] (default: {"localhost"})
      port {int} -- [Server port of the model.] (default: {9000})
    """

    self.model_name = model_name
    self.timeout = timeout
    self.host = host
    self.port = port

    channel = implementations.insecure_channel(self.host, self.port)
    self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  def _note_seq_to_text_seq(self, note_seq):
    
    # ### PIPELINE MAP ###
    # # Converts NoteSequence to TextSequence #
    #
    # DagInput > Quantizer > PerformanceExtractor > 'MetricPerformance'
    # DagInput > MetadataExtractor > 'metadata'
    # 
    # {'MetricPerformance', 'meta'} > ParserToText > DagOutput
    
    
    key = 'live'
    quantizer = Quantizer(
        steps_per_quarter=STEPS_PER_QUARTER, 
        name='Quantizer_' + key)
    perf_extractor = PerformanceExtractor(
        min_events=MIN_EVENTS,
        max_events=MAX_EVENTS,
        num_velocity_bins=0,
        name='PerformanceExtractor_' + key)
    meta_extractor = MetadataExtractor(
        name = 'MetadataExtractor' + key)
    parser = ParserToText(
        name='ParserToText' + key)

    dag = {}
    dag[quantizer] = dag_pipeline.DagInput(music_pb2.NoteSequence)
    dag[perf_extractor] = quantizer
    
    dag[meta_extractor] = dag_pipeline.DagInput(music_pb2.NoteSequence)
    
    dag[parser] = { 'MetricPerformance' : perf_extractor, 
                    'metadata' : meta_extractor }
    
    dag[dag_pipeline.DagOutput(key)] = parser

    # NoteSequence -> TextSequence
    text_seq = None
    pipeline = dag_pipeline.DAGPipeline(dag)
    output_names = pipeline.output_type_as_dict.keys()
    for name, outputs in _guarantee_dict(pipeline.transform(note_seq), list(output_names)[0]).items():
      for output in outputs:
        text_seq = output
        
    return text_seq

  def _text_seq_to_midi(self, model_output):
    
    result = TextSequence(model_output)
    note_seq = result.to_note_seq()

    output = tempfile.NamedTemporaryFile()
    magenta.music.midi_io.sequence_proto_to_midi_file(note_seq, output.name)
    output.seek(0)
    return output

  def _parse_inference(self, predicted):
    """Parses a binary translation result from Tensorflow.

    Args:
      predicted {`PredictResponse` proto} -- []

    Returns:
      A list of tokens. Each element in the list is a string.
    """

    hypothesis = []
    for event in predicted.outputs["tokens"].string_val:
      hypothesis.append(event.decode("utf-8"))
    
    return hypothesis

  def _infer(self, tokens):
    """Calls Tensorflow Serving to run inference.
      
      Returns: 
        A list of strings. Each element represents an event.
    """

    request = predict_pb2.PredictRequest()
    request.model_spec.name = self.model_name
    request.inputs["tokens"].CopyFrom(
        tf.make_tensor_proto([tokens], shape=(1, len(tokens))))
    request.inputs["length"].CopyFrom(
        tf.make_tensor_proto([len(tokens)], shape=(1,)))

    future = self.stub.Predict.future(request, self.timeout)

    predicted = future.result()
    predicted = self._parse_inference(predicted)
    predicted = ' '.join(predicted)

    return predicted

  def predict(self, user_note_seq, total_seconds=10):
    text_seq = self._note_seq_to_text_seq(user_note_seq)
    tokens = text_seq.split()
    result = self._infer(tokens)

    generated_midi = self._text_seq_to_midi(result)
    
    return generated_midi


user_sequence = [
    'SHIFT8 ON75 SHIFT6 OFF75 ON72 SHIFT2 OFF72 ON68 SHIFT4 OFF68 SHIFT8 ON70 SHIFT2 OFF70 ON72 SHIFT2 OFF72 ON70 SHIFT2 OFF70 ON70 SHIFT2 OFF70 ON70 SHIFT4 OFF70 ON70 SHIFT2 OFF70 ON68 SHIFT2 OFF68 ON70 SHIFT2 OFF70 ON68 SHIFT2 OFF68 SHIFT2 ON75 SHIFT4 OFF75 ON75 SHIFT2 OFF75 ON75 SHIFT4 OFF75 SHIFT4 ON77 SHIFT4 OFF77 ON75 SHIFT8 OFF75 ON73 SHIFT2 OFF73 ON72 SHIFT2 OFF72 ON68 SHIFT4 OFF68 SHIFT6 ON68 SHIFT2 OFF68 ON70 SHIFT2 OFF70 ON72 SHIFT2 OFF72 ON70 SHIFT2 OFF70 ON75 SHIFT2 OFF75 SHIFT2 ON70 SHIFT2 OFF70 ON70 SHIFT2 OFF70 ON68 SHIFT4 OFF68 ON75 SHIFT2 OFF75 ON75 SHIFT16 OFF75'
  ]