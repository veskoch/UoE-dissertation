# Arrangement of Popular Songs with Deep Learning #

This repository contains the code for the author's dissertation at the MSc Artificial Intelligence program at The University of Edinburgh. -- Summer 2018


### Abstract of Proposed Research ###
The proposed dissertation project is building on the recent momentum in applying deep learning to music-related tasks. Most of the academic and private sector efforts have gone to devise full algorithmic composition of new content imitating the training corpus. However, the present research takes a different approach. The author looks at building a technology which can support rather than supplant the human creator. Over 12 weeks in the summer, the author will have access to a never-before-used dataset from the music industry to design a Recurrent Neural Network- based architecture which can transform simple monotonic melodic cues from popular songs to elaborate compositions backed by chords, taking extreme ranges and varying rhythmically and dynamically – a reconceptualization known in music jargon as arrangement.


## Workflow ##
On a high level, the process is:

Collate > Preprocess > Build Vocab > Define Model > Train & Eval

Each of those steps comes with a set of possible configurations which are explored for the dissertation.

## Dependencies ##
The instruction below pertain to Mac OSX High Sierra.


### System-level ###
* Python == 3.6 (for `model`)
* Python == 2.7 (for `deploy`)
* Anaconda
* [fluidsynth](http://www.fluidsynth.org/) = 1.1.11
    * `brew install fluidsynth pkg-config`
    * FluidSynth is a software synthesizer for generating music. It's the software analogue of a MIDI synthesizer.  You load patches, set parameters, then send NOTEON and NOTEOFF events to play notes. Instruments are defined in SoundFonts, generally files with the extension SF2. FluidSynth can either be used to play audio itself, or you can call a function that returns chunks of audio data and output the data to the soundcard yourself.
    * This is not a `pip` install! There is a module in PyPI of the same name but it is not what we need. It's very confusing.
* [gsutil](https://cloud.google.com/storage/docs/gsutil_install)
    * You need `gsutil` to download the Alexander Holm Salamander piano SoundFont from a Magenta GCS bucket (591.9 MiB)
        * `gsutil -m cp gs://download.magenta.tensorflow.org/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /tmp/`

### Python Libraries ###
Besides the modules specified in `requirements.txt` you also need to manually install:
* [pyfluidsynth](https://github.com/nwhitehead/pyfluidsynth)
    * This module contains python bindings for FluidSynth.


### Getting started ###
1. Install all dependencies above
2. Process the data
3. [Build dataset -- convert MusicXML to NoteSequences](https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md)


#### Who do I talk to? ####
For more information/enquiries contact one the administrators of the repository.

#### Author ####
* Vesko Cholakov