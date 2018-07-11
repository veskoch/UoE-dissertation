import os
import re
import shutil

import pandas as pd

from magenta.music import musicxml_reader
from magenta.music import note_sequence_io 

from collections import Counter

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
    
    def __init__(self):
        self.src_folder = None
        self.files_index = None
        self.collated_index = None
        
    def build_index(self, src_folder, ext_meta=None):
        """ Builds DataFrame which indexes and classifies all files in `src_folder`."""

        self.src_folder = src_folder
        files_index = dict()

        if ext_meta is not None:
            ext_meta = pd.read_csv(ext_meta)

        for path, directories, files in os.walk(self.src_folder):
            for f in files:
                file_match = re.match(r'^[A-Za-z0-9]+(_)(adv|int|beg)-[A-Za-z0-9]+-(lh|rh|bh).xml$', f)
                if file_match:
                    file_id = f
                    name_id = re.match(r'^[A-Za-z0-9]+(?=(_))', f).group(0)
                    level = re.search(r'(?<=_)(adv|int|beg)(?=(-))', f).group(0)
                    segment = re.search(r'(?<=_(adv|int|beg)-)[A-Za-z0-9]+(?=(-))', f).group(0)
                    hand = re.search(r'(?<=-)(lh|rh|bh)(?=(.xml))', f).group(0)

                    files_index[file_id] = {
                        "name_id" : name_id,
                        "level" : level,
                        "segment" : segment,
                        "hand" : hand,
                        "path" : os.path.join(path, f)
                    }

                    if ext_meta is not None: # Look for additional metadata
                        meta_sliced = ext_meta[(ext_meta['ID'] == name_id) & (ext_meta['Level'] == level)]

                        if len(meta_sliced) == 0:
                            print('INFO: Skipped metadata for {}. Found no metadata.'
                                                                      .format(name_id))
                        elif len(meta_sliced) != 1:
                            print('INFO: Skipped metadata for {}. Found multiple conflicting rows.'
                                                                      .format(name_id))
                        else:
                            for key in [('key', 'Magenta_Key'),
                                         ('mode', 'MajorMinor'),
                                         ('time_sig', 'Time_Sig'),
                                         ('genre', 'Genre'),
                                         ('name', 'Name'),
                                         ('artist', 'Artist')]:
                                        
                                files_index[file_id][key[0]] = meta_sliced[key[1]].values[0]

        self.files_index = pd.DataFrame.from_dict(files_index, orient='index').sort_values(by=['segment'])
        
    def collate(self, 
                hand=('bh', 'bh'),
                level=[('int', 'adv'), ('beg', 'adv'), ('beg', 'int')],
                includeWholeSong=False,
                eval_set = [],
                test_set = []
               ):
        """Collates source -> target .xml pairs from the data in the `files_index` df. 
            
            Args:
                hand: A tuple such as ('lh', 'rh') which shows desired input and target hand. 
                A hand is one of `lh`, `rh`, `bh`.
                includeWholeSong: `False` includes all segments except wholeSong, `True` 
                    includes all segments including wholeSong
                level: A list of tuples, where the first element is the desired source
                    level of playing difficulty, and the second element is the target
            
            Returns:
                A list of ('input_xml_path', 'target_xml_path') tuples 
                of paths to .xml files. Both paths point to .xml files of the same song, 
                segment and hand but varying difficulty. The first element is the input
                musical composition from which we want to translate, and the second element 
                must be the target .xml composition to which we want to translate.
                
            Asserts validity of arguments.
            
        """
        assert set(sum(level, ())).issubset(('int', 'adv', 'beg'))
        assert set(hand).issubset(('lh', 'rh', 'bh'))
        assert len(hand) == 2
        level = set(level)
        
        collated = { 'train': [],
                     'eval': [],
                     'test': []
                     }

        counter_by_song = { 'train': Counter(),
                            'eval': Counter(),
                            'test': Counter()
                            }
        counter_by_level = { 'train': Counter(),
                             'eval': Counter(),
                             'test': Counter()
                             }
        counter_by_segment = { 'train': Counter(),
                               'eval': Counter(),
                               'test': Counter()
                             }
        
        # A few lines for efficiency
        # Slice the index to keep only the rows we will need
        if hand[0] == hand[1]:
            _songs_sliced_df = self.files_index.loc[self.files_index['hand'] == hand[0]]
        else:
            _songs_sliced_df = self.files_index.copy(deep=True)
        if not includeWholeSong:
            _songs_sliced_df = _songs_sliced_df.loc[_songs_sliced_df['segment'] != 'wholeSong']

        # Iterate over all songs
        for name_id in self.files_index['name_id'].unique():

            if name_id in eval_set:
                collection = 'eval'
            elif name_id in test_set:
                collection = 'test'
            else:
                collection = 'train'

            # Temp dataframe sliced by the current song and hand
            _song_df = _songs_sliced_df.loc[_songs_sliced_df['name_id'] == name_id]
            # Get available levels for a song
            available_levels = _song_df['level'].unique()

            for pairing in level:
                assert len(pairing) == 2
                # Check which requested pairings are possible
                if pairing[0] in available_levels and pairing[1] in available_levels:
                    src = _song_df.loc[(_song_df['level'] == pairing[0])
                                        & (_song_df['hand'] == hand[0])]
                    tgt = _song_df.loc[(_song_df['level'] == pairing[1])
                                        & (_song_df['hand'] == hand[1])]
                    try:
                        # Two levels of difficulty must have matching segments
                        assert list(src['segment']) == list(tgt['segment'])
                    except:
                        print('INFO: Skipping "{}" because of mismatching segments or hand parts.\nSource:\n{}\nTarget:\n{}'
                              .format(name_id, src['path'], tgt['path']))

                    # Collate
                    collated[collection] += list(zip(src['path'], tgt['path']))

                    # STATS
                    counter_by_song[collection].update({name_id: len(src)})
                    counter_by_level[collection].update({pairing: len(src)})
                    counter_by_segment[collection].update(src['segment'].values)

        print('\nINFO: Successfully collated Train: {} pairs from {} unique songs.'.format(len(collated['train']), len(counter_by_song['train']) ))
        print('INFO: Successfully collated Eval: {} pairs from {} unique songs.'.format(len(collated['eval']), len(counter_by_song['eval']) ))
        print('INFO: Successfully collated Test: {} pairs from {} unique songs.'.format(len(collated['test']), len(counter_by_song['test']) ))

        print('\nCount of pairs by level:')
        print(counter_by_level)
        print('\nCount of pairs by segment type:')
        print(counter_by_segment)
        print('\nCount of pairs by song:')
        print(counter_by_song)

        self.collated_index = collated
    
    def _xml_to_seq_proto(self, full_xml_path, seq_collection_name):
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

            sequence.collection_name = seq_collection_name
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

        # Iterate over 'train', 'eval' and 'test' collections
        for collection in self.collated_index.keys():

            inputs_path = os.path.join(target_dir, collection + '_inputs' + '.tfrecord')
            targets_path = os.path.join(target_dir, collection + '_targets' + '.tfrecord')

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
                for i, pair in enumerate(self.collated_index[collection]):
                    input_proto = self._xml_to_seq_proto(pair[0], collection + '_inputs')
                    target_proto = self._xml_to_seq_proto(pair[1], collection + '_targets')

                    inputs_writer.write(input_proto)
                    targets_writer.write(target_proto)

            print('INFO: Saved {}.'.format(inputs_path))
            print('INFO: Saved {}.'.format(targets_path))