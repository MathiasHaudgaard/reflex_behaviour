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

#Example
python drive.py models/model_50000

```

You also need to run car simulator in unity in test mode.

## Tips for training and inference
If you are stuck with a low end computer it's best to run the unity simulator with the lowest resolution. This will make it easier for the unity simulator to apply the steering commands in time before the car will be in another position. This happens since it's an asynchronous client/server relationship between the unity simulator and the ML model.

I collected data by driving around the lake track for 5 rounds. The model_50000, which you can find in the model folder, oversteers a bit but it stays within the track.

Feel free to leave an issue if you're having any problems :-)

## Credits

Credits to [naokishibuya](https://github.com/naokishibuya) and to the paper 'End to End Learning for Self-Driving Cars' from Bojarski et al.
