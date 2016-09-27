# Dimensionality reduction

## Exploring the face data
There is no built-in functionality to allow us to read images.
Spark provides a method called wholeTextFiles, which allows us to operate on entire files at once.

## Training
Each image can be represented as a three-dimensional array, or matrix, of pixels. 
The first two dimensions, that is the x and y axes, represent the position 
of each pixel, while the third dimension represents the red, blue, and 
green (RGB) color values for each pixel.
We convert to grayscale and resize images because image-processing tasks 
can quickly become extremely memory intensive.

## Input data
We use the Labeled Faces in the Wild (LFW) dataset of facial images. 
This dataset contains over 13,000 images of faces generally taken from 
the Internet and belonging to well-known public figures. In order to avoid 
to download a very big dataset we work with a subset of the images, 
using people who have names that start with an "A". This dataset can be 
downloaded from [here](http://vis-www.cs.umass.edu/lfw/lfw-a.tgz).