# Reflex behaviour

## Overview

This is an updated version of the repo https://github.com/naokishibuya/car-behavioral-cloning from naokishibuya which uses Pytorch instead of Keras.

## Dependencies

You can install all dependencies by running one of the following commands

You need a [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

```python

conda env create -f environments.yml

```

## Usage


### Start the simulator

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene and press the Training Mode.  Then, collect the images for training.

### Train the model

To train the model, call the following script:

```python
python train.py -d <data directory>
```
There are more arguments you can specify and you can see them all by calling:

```python
python train.py -h
```

### Test the model

All the models that will be created will be within the models directory which is created during training.

Test one of the models by running the the script:

```python

python drive.py <path to model_weights>

```

You also need to run car simulator in unity in test mode.

## Credits

Credits to [naokishibuya](https://github.com/naokishibuya) and to the paper 'End to End Learning for Self-Driving Cars' from Bojarski et al.
