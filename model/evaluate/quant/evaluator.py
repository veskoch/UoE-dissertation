import collections
import pandas as pd
import os

import sys
import inspect
from pathlib import Path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = Path(currentdir).parents[1]
sys.path.insert(0, os.path.join(parentdir, 'utils'))

from metrics import BLEU, ROUGE, MusicScores

"""Thin wrapper around evaluator.py for use in the Jupyter 1.4 Evaluation Notebook."""

class Evaluator:
    
    def __init__(self, targets, predicted):
        self.targets = targets
        self.predicted = predicted
        
        self._bleu = None
        self._rouge = None
        self._music_scores = None
        
    @property
    def bleu(self):
        if not self._bleu:
            b = BLEU()
            self._bleu = b.score(self.targets, self.predicted)
        print('INFO: BLEU done.')
        return {'bleu' : self._bleu}
        
    @property
    def rouge(self):
        if not self._rouge:
            r = ROUGE()
            self._rouge = r.score(self.targets, self.predicted)
        print('INFO: ROUGE done.')
        return self._rouge
    
    @property
    def music_scores(self):
        if not self._music_scores:
            ms = MusicScores()
            self._music_scores = ms.score(self.targets, self.predicted)
        print('INFO: MusicScores done.')
        return self._music_scores
           
    def get(self, to_csv_name=None):
        print('INFO: Crunching...')
        scores = { **self.bleu, **self.rouge, **self.music_scores }
        scores = pd.Series(scores,
                           index=[
                               'bleu',
                               'rouge-1_f', 'rouge-1_p', 'rouge-1_r',
                               'rouge-2_f', 'rouge-2_p', 'rouge-2_r',
                               'rouge-l_f', 'rouge-l_p', 'rouge-l_r',
                               'cc', 'cc_dist', 'tc', 'tc_dist',
                               'key_name_acc', 'key_mode_acc',
                               'tempo', 'tempo_dist',
                               'duration_dist',
                               'note_density_ratio'])
        
        if to_csv_name:
            save_dir = os.path.dirname(self.predicted)
            save_path = os.path.join(save_dir, to_csv_name)
            scores.to_csv(save_path)
            print('INFO: Saved as {}'.format(save_path))  

        return scores  