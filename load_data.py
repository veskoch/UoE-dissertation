import os
import re
import shutil

import pandas as pd

import constants

from magenta.music import musicxml_reader
from magenta.music import note_sequence_io 

class SourceFolderManager():
    """Manager for the folder which holds the raw .xml data files.
    
    Args:
        src_folder: Path to folder which holds .xml files. 
        tgt_folder: Path to folder where to save the reorganized files.
    
    Files inside `src_folder` must be organized by song, such that all 
        files for a song are in one folder. No two songs can be in the
        same folder.
    Files must follow the naming convention:
        [Song Name]_[Performance Level]-[Song Segment]-[Hand].xml, where:
            Performance Level is one of ['_beg', '_int', '_adv']
            Hand is one of ['lh', 'rh', 'bh']
            Song Segment must be a unique string given a song and a 
                performance level. Additionally, Song Segments must have 1:1
                correspondence across the Performance Levels of a song.
    Folders can be nested and do NOT need to follow a naming convention.
    
    """
    
    def __init__(self, 
                 src_folder):
        self.src_folder = src_folder
        self.files_index = None
        self.collated_index = None
        
    def build_index(self):
        """ Builds DataFrame which indexes and classifies all files in `src_folder`."""
        
        files_index = dict()
        
        for path, directories, files in os.walk(self.src_folder):
            for file in files:
                file_match = re.match(r'^[A-Za-z0-9]+(_)(adv|int|beg)-[A-Za-z0-9]+-(lh|rh|bh).xml$', file)
                if file_match:
                    file_id = file
                    name = re.match(r'^[A-Za-z0-9]+(?=(_))', file).group(0)
                    level = re.search(r'(?<=_)(adv|int|beg)(?=(-))', file).group(0)
                    segment = re.search(r'(?<=_(adv|int|beg)-)[A-Za-z0-9]+(?=(-))', file).group(0)
                    hand = re.search(r'(?<=-)(lh|rh|bh)(?=(.xml))', file).group(0)
                    
                    files_index[file_id] = {
                        "name" : name,
                        "level" : level,
                        "segment" : segment,
                        "hand" : hand,
                        "path" : os.path.join(path, file)
                    }
                    
        self.files_index = pd.DataFrame.from_dict(files_index, orient='index').sort_values(by=['segment'])
        
    def _collate_stats(self, collated):
        """ Return statistics on the collated list.
                - number of pairs total
                - number by type of pair
                - number of unique songs
                - histogram of pairs by level
                - histogram of pairs by segment
                - number of files ignored with another extension
                - size of collated .xml files

                TO-DO Save log file with a list of pairs which have been created.
                + Errors log
        """
        pass
        
    def collate(self, 
                hand='bh',
                includeWholeSong=False,
                level=[('int', 'adv'), ('beg', 'adv'), ('beg', 'int')],
                limit=float('-inf')
               ):
        """Collates source -> target .xml pairs from the data in the `files_index` df. 
            
            Args:
                hand: One of `lh`, `rh`, `bh`.
                includeWholeSong: `False` includes all segments except wholeSong, `True` 
                    includes all segments including wholeSong
                level: A list of tuples, where the first element is the desired source
                    level of playing difficulty, and the second element is the target
                limit: an int, the upper bound of the number of songs to process. Useful
                    for testing & debugging
            
            Returns:
                A list of ('input_xml_path', 'target_xml_path') tuples 
                of paths to .xml files. Both paths point to .xml files of the same song, 
                segment and hand but varying difficulty. The first element is the input
                musical composition from which we want to translate, and the second element 
                must be the target .xml composition to which we want to translate.
                
            Asserts validity of arguments.
            
        """
        assert set(sum(level, ())).issubset(('int', 'adv', 'beg'))
        level = set(level)
        assert hand in ['lf', 'rh', 'bh']
        
        collated = list()
        
        _songs_sliced_df = self.files_index.loc[self.files_index['hand'] == hand]
        if not includeWholeSong:
            _songs_sliced_df = _songs_sliced_df.loc[_songs_sliced_df['segment'] != 'wholeSong']
        
        # Iterate over all songs
        for i, song_name in enumerate(self.files_index['name'].unique()):
            if i > limit:
                break
            else:
                # Temp dataframe sliced by the current song and hand
                _song_df = _songs_sliced_df.loc[(_songs_sliced_df['name'] == song_name) & 
                                                (_songs_sliced_df['hand'] == hand)]
                # Get available levels for a song
                available_levels = _song_df['level'].unique()
                # Check which requested pairings are possible
                for pairing in level:
                    assert len(pairing) == 2
                    if pairing[0] in available_levels and pairing[1] in available_levels:
                        src = _song_df.loc[_song_df['level'] == pairing[0]]['segment']
                        tgt = _song_df.loc[_song_df['level'] == pairing[1]]['segment']
                        try:
                            # Two levels of difficulty must have matching segments
                            assert list(src) == list(tgt)
                            src = _song_df.loc[_song_df['level'] == pairing[0]]['path']
                            tgt = _song_df.loc[_song_df['level'] == pairing[1]]['path']
                            collated += list(zip(src, tgt))
                        except:
                            print('INFO: Skipping "{}" because of mismatching segments.'.format(song_name))
        
        print('INFO: Exported {} collated pairs.'.format(len(collated)))
        self.collated_index = collated
    
    def _xml_to_seq_proto(self, full_xml_path, collection):
        """Converts an individual .xml file to a NoteSequence proto. 
        Args:
            full_xml_path: the full path to the file to convert.
            collection: name of collection to which to save.
        Returns:
            NoteSequence proto or None if the file could not be converted.
        """
        
        if (full_xml_path.lower().endswith('.xml') or
            full_xml_path.lower().endswith('.mxl')):

            try:
                sequence = musicxml_reader.musicxml_file_to_sequence_proto(full_xml_path)
            except musicxml_reader.MusicXMLConversionError as e:
                print('INFO: Could not parse MusicXML file {}. It will be skipped. \
                      Error was: {}'.format(full_xml_path, e))
                return None

            sequence.collection_name = collection
            sequence.filename = os.path.basename(full_xml_path)
            sequence.id = os.path.basename(full_xml_path)
            return sequence
        
        else:
            print('INFO: Unable to find a converter for file {}'.format(full_xml_path))
    
    def serialize_collated(self, target_dir):
        """Saves to disk two .tfrecord files, each holding serialized NoteSequence protos
        from the previously collated list of .xml file paths.

        Args:
            target_dir: directory where to save the .tfrecord files
            
        """

        inputs_path = os.path.join(target_dir, 'inputs' + '.tfrecord')
        targets_path = os.path.join(target_dir, 'targets' + '.tfrecord')

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print('INFO: Created {} directory.'.format(target_dir))
        else:
            for path in [inputs_path, targets_path]:
                if os.path.exists(path):
                    raise FileExistsError('File {} already exists. Please remove and try again.'
                                        .format(path))           


        with note_sequence_io.NoteSequenceRecordWriter(inputs_path) as inputs_writer, \
        note_sequence_io.NoteSequenceRecordWriter(targets_path) as targets_writer:
            for i, pair in enumerate(self.collated_index):
                input_proto = self._xml_to_seq_proto(pair[0], 'inputs')
                target_proto = self._xml_to_seq_proto(pair[1], 'targets')

                inputs_writer.write(input_proto)
                targets_writer.write(target_proto)