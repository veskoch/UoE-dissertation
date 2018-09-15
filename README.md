# A.I. Enhancer #

This repository contains the code for the author's dissertation in MSc Artificial Intelligence at The University of Edinburgh in the summer of 2018.

The project shows for the first time an exciting AI implementation which generates and synchronizes appropriate accompaniment to musical melodies in different genres. The inputted melodies can be polyphonic, of varied length, varied difficulty and in different key and time signature. 

The workhorse of the model is a deep recurrent sequence-to-sequence architecture (Sutskever et al., 2014, Cho et al., 2014) which has not been used before in a musical context and is known for its successful applications to natural language processing tasks like question-answering, machine translation, image captioning, speech generation and text summarization.

The project would not have been possible without a unique training dataset which was generously provided by my friend Matt Entwistle at a New York-based educational company which offers online piano lessons. The dataset is a a collection of 200 popular commercial songs, each arranged for piano in at least two different level of performing difficulty (easy, intermediate, advanced) with separate tracks for the melody and the accompaniment. Unfortunately, the data is not part of this repository as the tracks are copyrighted.


### Context ###

The short timeline of the project necessitated the clever use open source libraries and the construction of comprehensive software architecture to automate the collating, preprocessing, training, testing and evaluation.  Following the initial stages of research and planning, I spent a month and a half incrementally developing, testing and refactoring the codebase while gathering feedback from musicians. I developed in Python in Visual Studio, version controlled on GitHub and trained the final models on GPU-accelerated VMs (NVIDIA Tesla K80/P100/V100) on Google Cloud. I developed a suite for automated evaluation and testing and deployed the final model with Flask, Node.js and TensorFlow Serving for interactive front-end serving via a web browser. 

### The Project ###

The project uses [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) which provides open source tools for sequence learning using TensorFlow. The [Magenta project](https://github.com/tensorflow/magenta), Google’s stab at creating art and music with machine learning, provided a large amount of useful methods for data handling and data preprocessing which I forked and customized for the current project.

The `model` directory holds of the code necessary to build the training dataset (collating the training files & preprocessing the content of the individual files, as well as defining, training and evaluating the model. A few evaluation methods are provided for both quantitative (ROUGE, BLEU, tonal certainty, correlation coefficient, key accuracy, note density) and qualitative evaluation (most notably an audio synthesizer and a tool for visualizing the generated music compositions as a pianoroll).

The `deploy` directory holds the files for front-end deployment, which are mostly borrowed from [AI Duet by Yotam Mann](https://github.com/googlecreativelab/aiexperiments-ai-duet), which provided a beautiful front-end adopted to demo the current project.

A PDF of the dissertation is available in the root directory of the project.

### System Requirements ###
* Python == 3.6 (for `model` directory)
* Python == 2.7 (for `deploy` directory)
* [Jupyter Notebook](jupyter.org/)
* [fluidsynth](http://www.fluidsynth.org/) = 1.1.11
    * `brew install fluidsynth pkg-config`
    * FluidSynth is a synthesizer for generating music (the software analogue of a MIDI synthesizer)  You load patches, set parameters, then send NOTEON and NOTEOFF events to play notes. Instruments are defined in SoundFonts with the extension .sf2.
    * This is not a `pip` install! There is a module in PyPI of the same name but it is not what we need.
* [gsutil](https://cloud.google.com/storage/docs/gsutil_install)
    * You need `gsutil` to download the Alexander Holm Salamander piano SoundFont from a Magenta GCS bucket (591.9 MiB)
    * `gsutil -m cp gs://download.magenta.tensorflow.org/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /tmp/`

### Python Dependencies ###
Besides the modules specified in `requirements.txt`, you also need to manually install [pyfluidsynth](https://github.com/nwhitehead/pyfluidsynth) which contains python bindings for FluidSynth.

### Who do I talk to? ###
For more information & inquiries contact Vesko Cholakov.


Last updated: September 2018