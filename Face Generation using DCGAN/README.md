<h1 align="center">Face Generation using DCGAN</h1>

## Project Outline:
In this project, a DCGAN is built and trained on a dataset of faces, and to get the generator network to generate new images of faces that look close to real faces.


The project is broken down into a series of following tasks:  
1. Pre-process and Loading data
2. Defining the Model
  2.1 Building Discriminator
  2.2 Building Generator
  2.3 Initialize the weights of the networks
  2.4 Building complete network
  2.5 Discriminator and Generator Losses
  2.6 Optimizers
3. Training
  3.1 Training Code
  3.2 Training
  3.3 Training Loss
  3.4 Generator samples from training
4. Analysis of Generated Images
  
* helper.py

## Dataset:

The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is used to train the GAN network. The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity colour(RGB) images with annotations.

## Result:

 
 ## Licensing, Authors, Acknowledgements
Credits must be given to Udacity for supporting while project development.

## Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Pytorch with CUDA](https://pytorch.org/)

