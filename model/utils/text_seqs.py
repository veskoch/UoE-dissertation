
from magenta.music.performance_lib import PerformanceEvent
import magenta.music as mm 

import re

class TextSequenceCollection():
    """Loads a collection of music text sequences and provides utils 
    to view, listen and export.
    
    Currently, only loading from file is supported. Sequences must be 
    separated by a new line character.
    
    Args:
        path: path to the file to load
        
    Returns:
        A list of music text sequences.
    
    """
    
    def __init__(self, path):
        self.text_seq_list = None
        self._as_note_seq = None
        self._as_midi = None
        
        with open(path, 'r') as f:
            collection = f.read().splitlines()
        self.text_seq_list = collection

    def __len__(self):
        return len(self.text_seq_list)
    
    @property
    def as_note_seq(self):
        if not self._as_note_seq:
            self._as_note_seq = self._to_note_seq()
        return self._as_note_seq

    @property
    def as_midi(self):
        if not self._as_midi:
            self._as_midi = self._to_midi()
        return self._as_midi
        
    def _to_note_seq(self):
        """ Converts `self.text_seq_list` to a list of NoteSequences.
        
        Returns:
            List of NoteSequence class types.
        """
        
        l = []
        for line in self.text_seq_list:
            text_seq = TextSequence(line)
            l.append(text_seq.to_note_seq())
    
        return l
        
    def _to_midi(self):
        """ Converts `self.text_seq_list` to a list of midi objects.
        Returns:
            The resulting list of midi objects.
        """
            
        l = []
        for note_seq in self.as_note_seq:
            midi_obj = mm.sequence_proto_to_pretty_midi(note_seq)
            l.append(midi_obj)
            
        return l
    
    def synth(self, i):
        """ Synthesizes audio from the sequence indexed at `i`.
        
        Args:
            seq_index: an integer indicating which sequence to 
                load (from a line number)
        """
        # Lower sound quality than `fluidsynth` with Yamaha C5 or other 
        # good SoundFont but good fallback if you don't have the SoundFont
        # mm.ntebook_utils.play_sequence(sequence)
            
        mm.play_sequence(self.as_note_seq[i], mm.midi_synth.fluidsynth,
                                sf2_path='../../assets/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2')
        
    def viz(self, i):
        """Builds a Bokeh JS pianoroll for the sequence indexed at `i`.
        
        Args:
            seq_index: an integer indicating which sequence to 
                load (from a line number)
        """
            
        mm.plot_sequence(self.as_note_seq[i])

class TextSequence():
    """Utilities for reading-in, synthesizing, vizing and 
    exporting musical text sequences.
    
    Args:
        text_sequence: A text-like string of events (i.e. "ON50 ON71 SHIFT50 OFF71")
    """
    
    def __init__(self, text_sequence):
        self.text_sequence = text_sequence
    
    def to_note_seq(self):
        """Converts `self.text_sequence` to a NoteSequence.
        
        Returns:
            The resulting NoteSequence.
        """    
        events = self.text_sequence.split()
        
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
                continue
                
            event_value = int(re.search(r'[0-9]+', event).group(0))
            
            event = PerformanceEvent(event_type, event_value)
            performance.append(event)

        note_seq = performance.to_sequence()
        
        return note_seq