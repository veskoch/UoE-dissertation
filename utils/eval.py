from opennmt.utils import evaluator

from magenta.music.performance_lib import PerformanceEvent
import magenta.music as mm

import re


class TextSequence():
    """Utilities for reading-in, synthesizing, vizing and 
    exporting musical text sequences."""
    
    def __init__(self):
        self.note_sequence = None
    
    def _parse(self, seq):
        """Converts music text sequence to NoteSequence.
        Args:
            A text-like string of events (i.e. "ON50 ON71 SHIFT50 OFF71")
        Returns:
            The text converted into a NoteSequence class type.
        """    
        events = seq.split()
        
        performance = mm.MetricPerformance(
          steps_per_quarter=4)
        
        for event in events:
            if re.match(r'^ON[0-9]+$', event):
                event_type = 1         
            elif re.match(r'^OFF[0-9]+$', event):
                event_type = 2
            elif re.match(r'^SHIFT[0-9]+$', event):
                event_type = 3
            else:
                raise ValueError('Unknown event type: %s' % event)
                
            event_value = int(re.search(r'[0-9]+', event).group(0))
            
            event = PerformanceEvent(event_type, event_value)
            performance.append(event)

        note_seq = performance.to_sequence()
        return note_seq
       
    def from_file(self, path, seq_index):
        """Reads a sequence from a text file and converts it to a NoteSequence.
        
        Args:
            path: path to the text file
            seq_index: an integer indicating which sequence to 
                load (from a line number)
        """

        with open(path, 'r') as f:
            text_seqs = f.readlines()
            text_seqs = [l.strip() for l in text_seqs]
        
        text_seq = text_seqs[seq_index]
        print('INFO: Found {} sequence examples in {}.'
              .format(len(text_seqs), path))
        # print('INFO: Loading sequence indexed at {}.'
        #       .format(seq_index))

        note_seq = self._parse(text_seq)
        
        self.note_sequence = note_seq
    
    def synth(self):
        # Lower sound quality than `fluidsynth` with Yamaha C5 or other 
        # good SoundFont but good fallback if you don't have the SoundFont
        # mm.ntebook_utils.play_sequence(sequence)
        
        mm.play_sequence(self.note_sequence, mm.midi_synth.fluidsynth,
                                sf2_path='./assets/Yamaha-C5-Salamander-JNv5.1.sf2')
        
    def viz(self):
        """Creates an interactive player for a note sequence."""
        mm.plot_sequence(self.note_sequence)
        
    def export(self, midi_path):
        """Export the sequence as MIDI."""
        
        mm.sequence_proto_to_midi_file(self.note_sequence, midi_path)
        
        ## We can invoke to_sequence() on a Performance which will convert it to NoteSequence Proto. 
        # mm.sequence_proto_to_midi_file(generated_sequence, midi_path) can take a sequence proto and 
        # turn it into a MIDI.
        
class Evaluator():
    """Provides interactive tools and calculates BLEU, ROUGE and F-1 scores.
    
    """

    def __init__(self,
                targets_path,
                predicted_path):
        
        self.targets_path = targets_path
        self.predicted_path = predicted_path
        
        self.bokehs = []
        self.synths = []
    
    def score(self):
        
        bleu_evaluator = evaluator.BLEUEvaluator()
        bleu_score = bleu_evaluator.score(self.targets_path, self.predicted_path)
        print('BLEU: \t\t {}'.format(bleu_score))
        
        rouge_evaluator = evaluator.ROUGEEvaluator()
        rouge_score = rouge_evaluator.score(self.targets_path, self.predicted_path)
        print('\nROUGE-1: \t {}'.format(rouge_score['rouge-1']))
        print('ROUGE-2: \t {}'.format(rouge_score['rouge-2']))
        print('ROUGE-L: \t {}'.format(rouge_score['rouge-l']))
        
        f1_score = 2 * (bleu_score * rouge_score['rouge-1']) / (bleu_score + rouge_score['rouge-1'])
        print('\nF1: \t\t {}'.format(f1_score))