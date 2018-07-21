## A.I. Enhancer

A piano that enhances your creations. Come up with a simple melody. Hear it better.

## About

This experiment lets you make music through machine learning. A neural network was trained on many MIDI examples and it learned about musical concepts, building a map of notes and timings. You just play a few notes, and see how the neural net responds. 

## CREDITS

Built by [Vesko Cholakov](https://github.com/cholakov) and trained on a collection of popular songs arranged by Matthew Entwistle. The front-end is borrowed from [AI Duet](https://github.com/cholakov/ai-duet/), a similar project created by [Yotam Mann](https://github.com/tambien), the Magenta and Creative Lab teams at Google. 

The backend, however, has been redesigned for scratch for my dissertation. The project uses [TensorFlow](https://tensorflow.org), [OpenNMT](https://github.com/OpenNMT/OpenNMT-tf) and open-source tools from [Magenta](https://magenta.tensorflow.org/) and [music21](https://github.com/cuthbertLab/music21) and more (see requirements).



## OVERVIEW

A.I. Ehancer is composed of two parts, the front-end which is in the `static` folder and the back-end which is in the `server` folder. The front-end client creates short MIDI files using the players's input which is sent to a [Flask](http://flask.pocoo.org/) server. The server takes that MIDI input, "enhances" it and returns it back to the client. 

## INSTALLATION

A.I. Enhancer only works with [Python 2.7](https://www.python.org/download/releases/2.7/) and it has been tested with Node v6.

To get started, create a new a Python environment with Conda and install all of the dependencies.

```bash
cd server
pip install -r requirements.txt
```
If it _did_ install successfully, you can run the server by typing:

```bash
python server.py
```

Then to build and install the front-end Javascript code, first make sure you have [Node.js](https://nodejs.org) 6 installed. And then install of the dependencies of the project and build the code by typing the following in the terminal: 

```bash
cd static
npm install
npm run build
```


You also need to start `tensorflow_model_server` for inference:

```
tensorflow_model_server --port=9000 --enable_batching=true --batching_parameters_file=$HOME/deploy/server/batching_parameters.txt --model_name=arranger --model_base_path=$HOME/deploy/server/model/
```

You can now play with A.I. Enhancer at [localhost:8080](http://localhost:8080).


## MIDI SUPPORT

The A.I. Enhancer supports MIDI keyboard input using [Web Midi API](https://webaudio.github.io/web-midi-api/) and the [WebMIDI](https://github.com/cotejp/webmidi) library. 

## PIANO KEYBOARD

The piano can also be controlled from your computer keyboard thanks to [Audiokeys](https://github.com/kylestetz/AudioKeys). The center row of the keyboard is the white keys.

## AUDIO SAMPLES

Multisampled piano from [Salamander Grand Piano V3](https://archive.org/details/SalamanderGrandPianoV3) by Alexander Holm ([Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)).

String sounds from [MIDI.js Soundfonts](https://github.com/gleitz/midi-js-soundfonts) generated from [FluidR3_GM.sf2](http://www.musescore.org/download/fluid-soundfont.tar.gz) ([Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)).

## LICENSE

Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
