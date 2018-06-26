# Arrangement of Popular Songs with Deep Learning #

This repository contains the code for the author's dissertation at the MSc Artificial Intelligence program at The University of Edinburgh. -- Summer 2018


### Abstract of Proposed Research ###
The proposed dissertation project is building on the recent momentum in applying deep learning to music-related tasks. Most of the academic and private sector efforts have gone to devise full algorithmic composition of new content imitating the training corpus. However, the present research takes a different approach. The author looks at building a technology which can support rather than supplant the human creator. Over 12 weeks in the summer, the author will have access to a never-before-used dataset from the music industry to design a Recurrent Neural Network- based architecture which can transform simple monotonic melodic cues from popular songs to elaborate compositions backed by chords, taking extreme ranges and varying rhythmically and dynamically – a reconceptualization known in music jargon as arrangement.



## Dependencies ##
The instruction below pertain to Mac OSX High Sierra.


### System-level ###
* Python >= 3.6
* [Bazel](https://docs.bazel.build/versions/master/install.html) latest version

### Python Libraries ###
* IPython
* Anaconda

* [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf)
* [magenta](https://github.com/tensorflow/magenta/blob/master/README.md) = 0.3.8

* tensorflow
* numpy >= 1.11.0 – just because, how do you do ML without it
* pandas >= 0.18.1
* scipy >= 0.18.1

* pillow >= 3.4.2
* bokeh >= 0.12.0
* intervaltree >= 2.1.0
* librosa >= 0.6.0
* matplotlib >= 1.5.3
* mido == 1.2.6
* mir_eval >= 0.4
* pretty_midi >= 0.2.6
* python-rtmidi
* wheel

* [music21](http://web.mit.edu/music21/) – for handling MusicXML
* [MuseScore](https://musescore.org/en) – extends music21 to allow import/export of MusicXML, and view/edit/export of musical data


### Getting started ###

1. Install all dependencies above
2. Process the data
3. [Build dataset -- convert MusicXML to NoteSequences](https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md)



### Useful ###
Magenta is a [Git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules).



#### Who do I talk to? ####
For more information/enquiries contact one the administrators of the repository.

#### Author ####
* Vesko Cholakov






INPUT_DIRECTORY=/Users/vesko/Google\ Drive/Docs/Education/Edinburgh/Classes/DISS/Data/In-use
SEQUENCES_TFRECORD=file.tfrecord

bazel run //magenta/scripts:convert_dir_to_note_sequences -- --input_dir=$INPUT_DIRECTORY --output_file=$SEQUENCES_TFRECORD --recursive


bazel-bin/magenta/scripts/convert_dir_to_note_sequences --input_dir=$INPUT_DIRECTORY --output_file=$SEQUENCES_TFRECORD

